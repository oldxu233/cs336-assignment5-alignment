# pytest -k test_tokenize_prompt_and_output
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase):
    """
    对提示和输出字符串进行分词, 并构建一个掩码, 标记response token值为 1, 其余填充为 0。

    Args:
        prompt_strs: List[str] —— 提示字符串列表。
        output_strs: List[str] —— 输出字符串列表。
        tokenizer: PreTrainedTokenizer —— 用于分词的分词器。

    Returns:
        dict[str, torch.Tensor]：
            设 prompt_and_output_lens 为各拼接后序列的长度列表，
            返回字典包含以下键：
            - input_ids: shape (batch_size, max(prompt_and_output_lens) - 1)
                         拼接后的 token 序列（去掉最后一个 token）
            - labels: shape 同 input_ids，为 input_ids 右移一位（即去掉第一个 token）
            - response_mask: shape 同 input_ids，响应 token 对应位置为 True，其余为 False
    """
    input_prompts_ids, output_ids = [], []
    for p in prompt_strs:
        p_id = tokenizer.encode(p, add_special_tokens=False)
        input_prompts_ids.append(torch.tensor(p_id))
    for o in output_strs:
        o_id = tokenizer.encode(o, add_special_tokens=False)
        output_ids.append(torch.tensor(o_id))
    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(input_prompts_ids, output_ids)]
    D_output = max(prompt_and_output_lens) - 1
    
    # padding
    paded_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    input_ids = []
    labels = []
    response_mask = []
    for p_id, o_id in zip(input_prompts_ids, output_ids):
        # [问题tokens] + [答案tokens] + [EOS]
        input_id = torch.cat((p_id, o_id, torch.tensor([tokenizer.eos_token_id])), dim=-1)
        response_m = torch.cat((torch.zeros_like(p_id).to(dtype=torch.bool), #False
                                torch.ones_like(o_id).to(dtype=torch.bool),  #True
                                torch.tensor([False])), dim=-1)              #False
        
        # 自回归模型每次预测下一个token
        # 输入：[token1, token2, token3]
        # 标签：[token2, token3, token4]
        slice_input_id = input_id[:-1]     # 去掉最后一个token
        slice_output_id = input_id[1:]     # 去掉第一个token
        slice_response_m = response_m[1:]  # 响应掩码对应右移
        
        # 填充到统一长度
        pad_len = D_output - len(slice_input_id)
        padded_input_id = F.pad(slice_input_id, (0, pad_len), value=paded_val)
        padded_output_id = F.pad(slice_output_id, (0, pad_len), value=paded_val)
        response_mask_padded = F.pad(slice_response_m, (0, pad_len), value=False)

        input_ids.append(padded_input_id)
        labels.append(padded_output_id)
        response_mask.append(response_mask_padded)
    return {
        "input_ids": torch.stack(input_ids), # 输入序列 [batch_size, seq_len-1]
        "labels": torch.stack(labels),       # 目标序列 [batch_size, seq_len-1]
        "response_mask": torch.stack(response_mask)  # 布尔掩码 [batch_size, seq_len-1]
    }
    

    
