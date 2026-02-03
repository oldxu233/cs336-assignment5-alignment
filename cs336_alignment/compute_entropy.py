import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
	"""功能：获取下一个 token 预测的熵（即在词汇表维度上的熵）。
	参数：
	- logits: torch.Tensor，形状为 (batch_size, sequence_length, vocab_size)，包含未归一化的 logits。
	
	返回值：
	- torch.Tensor，形状为 (batch_size, sequence_length)，表示每个下一个 token 预测的熵。"""
	# log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
	# return -torch.sum(torch.exp(log_p) * log_p, dim=-1)
	log_p = torch.nn.functional.log_softmax(logits, dim=-1)
	p = torch.exp(log_p)
	ce = -torch.sum(p * log_p, dim=-1)
	return ce