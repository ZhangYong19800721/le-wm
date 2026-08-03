# 训练入口：准备数据与归一化变换，构建 LeJEPA 模型并启动 Lightning 训练。
import os
from functools import partial
from pathlib import Path

# 训练框架、配置系统和项目依赖。
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    # history_size 决定预测器上下文长度，num_preds 决定监督目标的时间偏移。
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    # 序列边界缺失的动作不能参与数值运算，因此以零动作填充。
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # 一次编码得到视觉潜在状态和对应的动作嵌入。
    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    # 截取历史上下文，作为条件预测器的输入。
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    # 目标从 n_preds 之后开始，与预测序列在时间上对齐。
    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    # 总损失由潜在空间预测误差和高斯分布正则项加权组成。
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    # 仅记录损失项，并在分布式训练进程之间同步日志。
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################
    # 将 Hydra 数据集配置解析为普通容器，再单独取出数据集名称。

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    # 如环境变量已设置，则复用本地数据缓存目录。
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    # 图像列先转换、归一化并缩放到模型输入尺寸。
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        # 为每个非图像数值列拟合独立的 z-score 标准化变换。
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        # frameskip 会拼接多个动作，因此动作编码器输入宽度需同步放大。
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    # 按顺序组合所有变换，并挂载到数据集读取流程。
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # 固定随机生成器使训练/验证划分和训练批次顺序可复现。
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # 训练集丢弃最后一个不完整批次；验证集保留全部样本。
    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################
    # 根据 Hydra 配置实例化 JEPA 世界模型。

    world_model = hydra.utils.instantiate(cfg.model)

    # stable_pretraining 使用模块名到优化器配置的映射管理训练参数。
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    # 将数据加载器封装为数据模块，并把自定义前向函数绑定到训练模块。
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################
    # 每个实验可通过 subdir 写入独立的检查点子目录。

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    # 仅在配置启用时创建 Weights & Biases 日志记录器。
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    # 保存解析后的模型配置，供后续加载和复现实验使用。
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # 回调在指定 epoch 间隔导出可直接加载的预训练权重。
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    # Lightning Trainer 负责设备、分布式执行、日志与训练循环。
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    # 若已有 Lightning 检查点则断点续训，否则从头开始。
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    # Manager 统一连接训练器、模型、数据与可选检查点并执行训练。
    manager()
    return


if __name__ == "__main__":
    # 仅在直接执行脚本时启动 Hydra 训练入口。
    run()
