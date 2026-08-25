# 评估入口：加载数据集与策略，在 MuJoCo 世界中执行规划评估并保存指标。
import os

# 指定无窗口的 EGL 渲染后端，便于在服务器或容器中运行 MuJoCo。
os.environ["MUJOCO_GL"] = "egl"

# 标准库依赖。
import time
from pathlib import Path

# 第三方依赖：配置管理、数值计算、模型推理与数据预处理。
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    # 将原始图像转换为张量，按 ImageNet 统计量归一化，并缩放到评估尺寸。
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def sample_eval_starts(episode_lengths, goal_offset_steps, num_eval, seed):
    """Uniformly sample valid ``(episode, start_step)`` pairs."""
    episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    valid_counts = np.maximum(episode_lengths - goal_offset_steps, 0)
    num_valid = int(valid_counts.sum())

    if num_valid < num_eval:
        raise ValueError(
            f"Not enough valid starting points: requested {num_eval}, "
            f"but only {num_valid} are available."
        )

    # Sample flat valid-start IDs, then map them back to episode-local steps.
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(num_valid, size=num_eval, replace=False))
    cumulative_counts = np.cumsum(valid_counts)
    episodes = np.searchsorted(cumulative_counts, selected, side="right")
    previous_counts = np.where(
        episodes == 0, 0, cumulative_counts[episodes - 1]
    )
    starts = selected - previous_counts
    return episodes, starts


def get_dataset(cfg, dataset_name):
    # 根据磁盘格式选择读取器；当前训练与评估数据使用 Lance。
    return swm.data.load_dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cfg.cache_dir,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    # 规划使用的总动作步数不能超过单次评估预算。
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    # 留出两倍评估预算作为环境回合上限，避免评估过程被提前截断。
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    # 当前观测和目标图像使用完全相同的预处理流程。
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    # 读取评估数据，并用完整统计数据拟合各数值列的标准化器。
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)

    # 为动作和其他低维状态列分别建立零均值、单位方差的处理器。
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        # 去除包含 NaN 的边界样本，防止污染均值和方差。
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            # 目标状态列复用对应观测列的标准化参数。
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    # 配置未显式给出策略时，默认采用随机策略作为基线。
    policy = cfg.get("policy", "random")

    if policy != "random":
        # 加载冻结的预训练世界模型，并据此构建规划求解器和策略。
        model = swm.wm.utils.load_pretrained(
            cfg.policy, cache_dir=cfg.cache_dir
        )
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        # 随机策略不需要模型、求解器或数据预处理器。
        policy = swm.policy.RandomPolicy()

    # 学习策略的结果写到模型缓存目录旁；随机基线写到当前脚本目录。
    results_path = (
        Path(
            swm.data.utils.get_cache_dir(
                cfg.cache_dir, sub_folder="checkpoints"
            ),
            cfg.policy,
        ).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    # 直接依据每回合长度构造有效起点，避免依赖格式特有的索引列。
    eval_episodes, eval_start_idx = sample_eval_starts(
        episode_lengths=dataset.lengths,
        goal_offset_steps=cfg.eval.goal_offset_steps,
        num_eval=cfg.eval.num_eval,
        seed=cfg.seed,
    )
    print(len(eval_episodes), "starting points sampled for evaluation.")

    # 将选定策略装入环境，随后开始实际评估。
    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    # evaluate 会返回指标，并按配置将评估视频写入结果目录。
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    # 追加写入本次配置、指标和耗时，保留同一文件中的历史运行结果。
    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs
        # 先记录完整配置，再记录本次评估结果，便于复现实验。

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    # 仅在直接执行脚本时启动 Hydra 评估入口。
    run()
