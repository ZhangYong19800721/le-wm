"""JEPA Implementation"""
# JEPA 世界模型：编码视觉观测与动作，并在潜在空间中预测未来状态。

# 深度学习与张量重排依赖。
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    # 张量需要切断梯度并复制；非张量配置值则原样返回。
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    # 组合视觉编码器、动作编码器和自回归预测器的核心模型。

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        # projector 与 pred_proj 可用于对齐编码器和预测器的特征空间。
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        # 输入像素形状为 (B, T, ...)，先合并批次与时间维以批量编码。
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        # 允许视觉 Transformer 对不同分辨率的位置编码进行插值。
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        # 投影后恢复 (B, T, D) 的时序特征布局。
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # 动作序列独立编码，供条件预测器使用。
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        # 预测器在动作条件下生成潜在状态，再投影到目标特征空间。
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################
    # 以下方法仅用于规划和推理，不参与训练阶段的前向损失计算。

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        # H 为已有观测历史长度；B、S、T 分别表示批次、候选方案和规划时域。
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        # 初始动作与已有观测对齐，剩余动作供自回归展开使用。
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        # 每个批次只需编码一份初始历史，再扩展到所有候选动作方案。
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        # 合并 B 和 S 后可用一次模型调用并行处理全部候选方案。
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        # 每轮仅保留最近 HS 步上下文，并把新预测追加到潜在状态序列。
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        # 所有候选动作消费完后再预测一次终态，使状态数比动作数多一。
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        # 恢复候选方案维度，供后续逐方案计算规划代价。
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        # predicted_emb 与 goal_emb 均保留批次和候选方案两个前导维度。
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        # 将最终目标特征广播到整条预测轨迹的形状。
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        # 规划只比较最终预测状态，并对时间/特征等后续维度求和。
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        # 代价计算要求输入中提供目标观测。
        assert "goal" in info_dict, "goal not in info_dict"

        # 将输入字典中的所有张量迁移到模型所在设备。
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # 从每个候选方案取同一份目标，并将 goal 像素适配为编码器的 pixels 键。
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        # 去掉 goal_ 前缀，使目标状态字段与普通观测字段使用相同编码路径。
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        # 目标编码不依赖动作；编码结果用于评价每条候选轨迹的终态。
        goal.pop("action")
        goal = self.encode(goal)

        # 展开所有候选动作，并返回每个批次、每个候选方案对应的代价。
        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
