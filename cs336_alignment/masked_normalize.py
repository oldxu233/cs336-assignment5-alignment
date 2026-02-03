# pytest -k test_masked_normalize
import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """对指定维度求和并通过常数归一化，仅考虑掩码中值为1的元素。

    参数：
    - tensor：torch.Tensor，需求和并归一化的张量。
    - mask：torch.Tensor，与tensor形状相同；值为1的位置会被纳入求和范围。
    - normalize_constant：float，用于归一化的除数常数。
    - dim：int | None，归一化前要求和的维度；若为None，对所有维度求和。

    返回值：
    - torch.Tensor，归一化后的和，其中掩码元素（mask == 0）不参与求和。"""
    sum_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(sum_tensor, dim=dim) / normalize_constant