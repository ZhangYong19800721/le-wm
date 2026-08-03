# 数据预处理与模型检查点保存所需的通用工具。
import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    # 使用 ImageNet 统计量转换图像，再调整到模型要求的方形尺寸。
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        # 均值和标准差由训练数据列预先计算并随对象一起序列化。
        self.mean = mean
        self.std = std

    def __call__(self, x):
        # 将输入标准化为近似零均值、单位方差，并统一转换为浮点张量。
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    # 读取整列数据并转换为 Torch 张量，以便计算统计量。
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    # 序列边界可能含 NaN，计算均值和标准差前必须排除这些行。
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    # 包装为按 source 读取、向 target 写回的数据集变换。
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        # run_name 决定导出目录，cfg 保存重建模型所需的结构配置。
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        # 仅由全局主进程写文件，避免分布式训练时产生重复或竞争写入。
        if trainer.is_global_zero:
            # 按配置间隔保存，同时确保最后一个 epoch 必定保存。
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        # 延迟导入可减少仅使用预处理工具时的模块加载开销。
        from stable_worldmodel.wm.utils import save_pretrained
        # 文件名带 epoch 编号，便于比较和回滚不同训练阶段的权重。
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
