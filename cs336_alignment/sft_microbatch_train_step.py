import torch
from .masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    参数:
	- policy_log_probs:形状为(batch_size, sequence_length),来自待训练监督微调(SFT)策略的逐token对数概率。
	- response_mask:形状为(batch_size, sequence_length),响应token对应位置为1,提示词/填充token对应位置为0。
	- gradient_accumulation_steps:每个优化器步骤对应的微批次数量。
	- normalize_constant:用于除法归一化的常数,默认设为1.0即可。
	
	返回值:
	- tuple[torch.Tensor, dict[str, torch.Tensor]]:
	  - loss:标量张量,微批次损失(已根据梯度累积进行调整),返回该值用于日志记录。
	  - metadata:字典,包含底层损失调用的元数据及其他需记录的统计信息。
    """
    cross_entropy = masked_normalize(tensor=policy_log_probs, mask=response_mask, dim=-1, 
                                     normalize_constant=normalize_constant)
    loss = -cross_entropy.mean(dim=-1)
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, {}