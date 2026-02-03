import torch
from typing import Callable, Literal
from .masked_normalize import masked_normalize

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """为每组 rollout 响应计算奖励,并按组进行归一化。

    参数：
    reward_fn: Callable[[str, str], dict[str, float]]  
        用于将 rollout 响应与标准答案（ground truth）进行比较并打分的函数,返回一个字典,
        包含键 "reward"、"format_reward" 和 "answer_reward"。
    
    rollout_responses: list[str]  
        策略生成的 rollout 响应列表。该列表长度为 rollout_batch_size,
        即 rollout_batch_size = n_prompts_per_rollout_batch * group_size。
    
    repeated_ground_truths: list[str]  
        每个样本对应的标准答案列表。该列表长度也为 rollout_batch_size,
        因为每个问题的标准答案被重复了 group_size 次（与每个问题对应的多个响应对齐）。
    
    group_size: int  
        每个问题（即每组）生成的响应数量。
    
    advantage_eps: float  
        用于归一化时避免除零的小常数。
    
    normalize_by_std: bool  
        若为 True,则用每组奖励的标准差进行归一化（即减去均值后除以标准差）；
        否则仅减去组内均值。

    返回：
    tuple[torch.Tensor, torch.Tensor, dict[str, float]]
        - advantages: shape (rollout_batch_size,),每条 rollout 响应的组内归一化奖励（即优势值）。
        - raw_rewards: shape (rollout_batch_size,),每条 rollout 响应的原始未归一化奖励。
        - metadata: 用户自定义的其他统计信息,可用于日志记录（例如奖励的均值、标准差、最大/最小值等）。
    """
    rewards = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(rollout_response, ground_truth)
        rewards.append(reward_dict["reward"])
    rewards = torch.tensor(rewards)         # [n_prompts_per_rollout_batch * group_size, 1]
    group = rewards.reshape(-1, group_size) # [n_prompts_per_rollout_batch, group_size]
    group_mean = torch.mean(group, dim=-1, keepdim=True) # [n_prompts_per_rollout_batch, 1]
    advantages = group - group_mean         # [n_prompts_per_rollout_batch, group_size]
    
    if normalize_by_std:
        group_std = torch.std(group, dim=-1, keepdim=True) # [n_prompts_per_rollout_batch, 1]
        advantages /= (group_std + advantage_eps)
    advantages = advantages.flatten() # [n_prompts_per_rollout_batch * group_size, 1]
    metadata = {
        'mean': torch.mean(rewards),
        'std': torch.std(rewards),
        'max': torch.max(rewards),
        'min': torch.min(rewards)
    }
    return advantages, rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个token的策略梯度损失,其中raw_rewards_or_advantages可为原始奖励或已归一化的优势值
    
    参数：
        raw_rewards_or_advantages: 形状为(batch_size, 1)的张量,每个滚动响应的标量奖励/优势值
        policy_log_probs: 形状为(batch_size, sequence_length)的张量,每个token的对数概率
    
    返回：
        形状为(batch_size, sequence_length)的张量,逐token策略梯度损失（将在训练循环中跨批次和序列维度聚合）
    """
    return - raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """    
    参数：
        advantages: 形状为(batch_size, 1)的张量,每个样本的优势值A
        policy_log_probs: 形状为(batch_size, sequence_length)的张量,待训练策略的逐token对数概率
        old_log_probs: 形状为(batch_size, sequence_length)的张量,旧策略的逐token对数概率
        cliprange: 裁剪参数ε 例如0.2
    
    返回：
        loss: 形状为(batch_size, sequence_length)的张量,逐token裁剪损失
        metadata: 用于记录你希望日志中保存的任意信息。我们建议记录每个 token 是否被裁剪（clipped）,即：在最小值操作（min）右侧的裁剪后策略梯度损失是否低于左侧的未裁剪损失。
    """
    ratio = torch.exp(policy_log_probs - old_log_probs) # [batch_size, sequence_length]
    
    left = ratio * advantages                # [batch_size, sequence_length]
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    right = clipped_ratio * advantages
    loss = -torch.min(left, right)          # [batch_size, sequence_length]
    clipped = right < left                  # [batch_size, sequence_length] bool
    metadata = {"clipped": clipped}
    return loss, metadata

def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """    
    无裁剪的GRPO损失函数
    """
    ratio = torch.exp(policy_log_probs - old_log_probs) # [batch_size, sequence_length]
    loss = -ratio * advantages                          # [batch_size, sequence_length]
    clipped = torch.zeros_like(loss, dtype=torch.bool)
    metadata = {"clipped": clipped}
    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    advantages: torch.Tensor | None = None,
    raw_rewards: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for 'no_baseline'"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for 'grpo_clip'"
        assert old_log_probs is not None, "old_log_probs is required for 'grpo_clip'"
        assert cliprange is not None, "cliprange is required for 'grpo_clip'"
        # return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for 'reinforce_with_baseline'"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor, # [batch_size, sequence_length]
    mask: torch.Tensor,   # [batch_size, sequence_length]
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    mean = torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)
    return mean

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, advantages, raw_rewards, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask)
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def grpo_microbatch_train_step_masked_normalize(
    policy_log_probs: torch.Tensor, # shape = (micro_train_batch_size, seq_len)
    response_mask: torch.Tensor,    # shape = (micro_train_batch_size, seq_len)
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None, # shape = (micro_train_batch_size, 1)
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, advantages, raw_rewards, old_log_probs, cliprange)
    constant = response_mask.sum(dim=-1).max().clamp(min=1).item()
    loss = masked_normalize(tensor=loss, mask=response_mask, dim=-1, normalize_constant=constant).mean()
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, metadata
