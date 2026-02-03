# pytest -k test_get_response_log_probs
import torch
import torch.nn.functional as F
from .compute_entropy import compute_entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """参数：
    - model：PreTrainedModel，用于评分的HuggingFace模型（若无需计算梯度，需放置在正确设备上并处于推理模式）。
    - input_ids：torch.Tensor，形状为（batch_size, sequence_length），由分词方法生成的拼接后的提示词+响应token。
    - labels：torch.Tensor，形状为（batch_size, sequence_length），由分词方法生成的标签。
    - return_token_entropy：bool，若为True，通过调用`compute_entropy`额外返回逐token熵。

    返回值：
    - dict[str, torch.Tensor]：
    - "log_probs"：形状为 [batch_size, sequence_lengt] 条件对数概率 log p_{\theta}(x_t | x_{<t})。
    - "token_entropy"可选：形状为[batch_size, sequence_length], 每个位置的逐token熵(仅当return_token_entropy=True时存在)
    """
    logits = model(input_ids).logits     # [batch, seq, vocab]
    log_probs = F.log_softmax(logits, dim=-1)              # [batch, seq, vocab]

    # 将 labels 中 -100 替换为 0（避免 gather 报错），但后续 mask 掉
    labels_clamped = labels.clamp(min=0)
    # gather log prob for true tokens
    log_probs = torch.gather(log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)  # [batch, seq]
    mask = (labels != -100).float() # -100是padding位， mask: 1 表示有效位置，0 表示忽略位置
    log_probs = log_probs * mask

    if return_token_entropy is True:
        token_entropy = compute_entropy(logits=logits)
        token_entropy = token_entropy * mask
    else:
        token_entropy = None

    return {"log_probs": log_probs,
            "token_entropy": token_entropy}