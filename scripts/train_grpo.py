"""
CUDA_VISIBLE_DEVICES=0,2 python scripts/train_grpo.py --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B" --output-path "grpo_logs/" --loss-type "reinforce_with_baseline" --train-samples 7473

CUDA_VISIBLE_DEVICES=0,2 python scripts/train_grpo.py --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B" --output-path "grpo_logs/" --loss-type "grpo_clip" --train-samples 7473
"""

import torch
import wandb 
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from unittest.mock import patch
import random
import json
from typing import Callable, List, Dict, Any, Literal
import argparse
import logging
import sys
import re
import time
import os
from copy import deepcopy

from cs336_basics.nanochat.common import DummyWandb, compute_cleanup
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.grpo import grpo_microbatch_train_step, compute_group_normalized_rewards, masked_mean, grpo_microbatch_train_step_masked_normalize
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

logger = logging.getLogger(__name__)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
train_data_path = "sft_logs/gsm8k_QwenMath_train_out.json"
test_data_path = "data/gsm8k/test.jsonl"
seed = 42
train_device = 'cuda:0'
vllm_device = "cuda:1"

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.6):
    """
    启动推理过程，此处使用vLLM将模型部署在与策略模型不同的GPU上。
    """
    vllm_set_random_seed(seed)
    # 从TRL借鉴的Monkeypatch：https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # 对vLLM进行补丁，确保：
    # （1）将vLLM模型部署到指定设备（world_size_patch）；
    # （2）跳过不适合当前场景的测试（profiling_patch）。
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    return llm

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    从https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670复制
    """
    policy.eval()
    policy.tie_weights()
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    torch.cuda.synchronize(torch.device(vllm_device))
    policy.train()


def to_float_(x):
     if isinstance(x, torch.Tensor):
          return x.float().item()
     elif isinstance(x, str):
          return float(x.strip())
     return float(x)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    

    overview = {"correct":0, "format_wrong":0, "answer_wrong":0, "count":0}
    for generated_text, answer in zip(generated_texts, answers):
        extracted_answer = extract_answer(answer)
        reward_dict = reward_fn(generated_text, extracted_answer)
        
        overview["count"] += 1
        if reward_dict["reward"] == 1:
            overview["correct"] += 1
        elif reward_dict["format_reward"] == 1:
            overview["answer_wrong"] += 1
        elif reward_dict["answer_reward"] == 1:
            overview["format_wrong"] += 1
    return overview

def load_dataset_and_format_qa(train_sample):
    train_data = load_train_dataset(train_data_path)        # list[{"prompt": "A conversation between User and...", "answer": " We ..."}, ..., {}]
    train_data = random.sample(train_data, min(len(train_data), train_sample))
    test_data = load_test_dataset(test_data_path)           # list[{"question": "1+1=?", "answer": "2"}, ..., {}]
    test_data = format_test_dataset(test_data, prompt_path) # list[{"prompt": "A conversation between User and Assistant... 1+1=?", "answer": "2"}, ...,{}]
    return train_data, test_data

def format_test_dataset(test_data, prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    formated_qa = []
    for d in test_data:
        pair = {
            "prompt": prompt_template.format(question = d["question"]),
            "answer": d["answer"]
        }
        formated_qa.append(pair)
    return formated_qa


def load_test_dataset(file_path):
    "加载jsonl文件"
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_train_dataset(file_path):
    "加载json文件"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa = []
    for item in data["results"]:
        qa.append({"prompt": item["prompt"], "answer": item["answer"]})
    return qa


def train_grpo_experiment(
        model, tokenizer, vllm, output_path, 
        train_sample: int,              # 从数据集中加载的样本数
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
        n_grpo_steps: int = 200,        # GRPO外层迭代次数
        learning_rate: float = 2e-5, 
        advantage_eps: float = 1e-6,
        rollout_batch_size: int = 256,  
        group_size: int = 8,            # 一个prompt例子，vllm采样出来group_size个response
        sampling_temperature: float = 1.0,
        sampling_min_tokens: int = 4,   # 参考 Expiter，禁止空字符串响应
        sampling_max_tokens: int = 1024,
        epochs_per_rollout_batch: int = 1,  # 在线策略（On-policy）
        train_batch_size: int = 256,        # 在线策略
        gradient_accumulation_steps: int = 256,  # microbatch 大小为 1，可在 4090D 显卡上运行; 
        use_std_normalization: bool = True,
        cliprange=0.2,
):
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size 必须能被 gradient_accumulation_steps 整除"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0, "rollout_batch_size 必须能被 group_size 整除"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size #  = 256 / 8 = 32

    assert train_batch_size >= group_size, "train_batch_size 必须大于或等于 group_size"

    # n = rollout_batch_size // micro_train_batch_size = 256 / 1
    num_iterations = n_grpo_steps * epochs_per_rollout_batch * (rollout_batch_size // micro_train_batch_size) // gradient_accumulation_steps
                        
    # Compute init
    autocast_ctx = torch.amp.autocast(device_type=train_device, dtype=torch.bfloat16)
    synchronize = torch.cuda.synchronize
    get_max_memory = torch.cuda.max_memory_allocated

    # wandb logging init
    # run = exp_name = f"grpo_gsm8k_lr={learning_rate}_{loss_type}_masked_normalize_on_QwenMath_" + time.strftime('%Y%m%d_%H%M%S') # wandb 运行名称（"dummy" 表示不使用 wandb 日志）
    # run = exp_name = f"n_grpo_steps={n_grpo_steps}_epochs_per_rollout_batch={epochs_per_rollout_batch}_train_batch_size={train_batch_size}_{loss_type}_on_QwenMath_" + time.strftime('%Y%m%d_%H%M%S')
    run = exp_name = f"n_grpo_steps={n_grpo_steps}_epochs_per_rollout_batch={epochs_per_rollout_batch}_train_batch_size={train_batch_size}_grpo_noclip_on_QwenMath_" + time.strftime('%Y%m%d_%H%M%S')
    # run = exp_name = f"n_grpo_steps={n_grpo_steps}_epochs_per_rollout_batch={epochs_per_rollout_batch}_train_batch_size={train_batch_size}_{loss_type}_QuestionOnlyPrompt_on_QwenMath_" + time.strftime('%Y%m%d_%H%M%S')

    use_dummy_wandb = run == "dummy"
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="cs336-grpo", name=run)
    
    # train
    grad_clip = 1.0               # 梯度裁剪阈值（L2 范数上限）
    # eval
    log_steps = 16
    eval_steps: int = 16

    adamw_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    
    trainstep = 0
    evalstep = 0
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    train_data, test_prompt = load_dataset_and_format_qa(train_sample)

    eval_sampling_params = SamplingParams(temperature=sampling_temperature, top_p=1.0, max_tokens=1024, min_tokens=sampling_min_tokens,
                stop=["</answer>"], include_stop_str_in_output = True)
    rollout_sampling_params = SamplingParams(temperature=sampling_temperature, top_p=1.0, max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens, n=group_size, stop=["</answer>"], include_stop_str_in_output = True, seed=seed)
    
    for grpo_step in range(n_grpo_steps):
        # 从问题集D中采样一批问题Db
        rollout_dataset = random.sample(train_data, n_prompts_per_rollout_batch) # shape = n_prompts_per_rollout_batch
        rollout_prompts = [data["prompt"] for data in rollout_dataset]           # shape = n_prompts_per_rollout_batch
        rollout_answers = [data["answer"] for data in rollout_dataset]           # shape = n_prompts_per_rollout_batch

        old_model = deepcopy(model)
        old_model.eval()
        load_policy_into_vllm_instance(old_model, vllm)
        # 对Db中的每个问题q，从πθold(· | q)中采样G个输出{o(i)}G i=1
        outputs = vllm.generate(rollout_prompts, rollout_sampling_params)  # outputs.shape = n_prompts_per_rollout_batch * group_size = rollout_batch_size
        
        # 提取出所有的response，构造prompts，responses和repeated_ground_truths列表
        repeated_ground_truths = []
        responses = []
        prompts = []
        for i, (prompt, answer) in enumerate(zip(rollout_prompts, rollout_answers)):
            extracted_answer = extract_answer(answer)
            for j in range(group_size):
                repeated_ground_truths.append(extracted_answer)
                prompts.append(prompt)
                generated_text = outputs[i].outputs[j].text
                responses.append(generated_text)
                # print(f"generated_text: {generated_text}")
        # 计算每个o的奖励r 和 advantage
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, responses, repeated_ground_truths,
                                                                             group_size, advantage_eps, use_std_normalization)
        
        advantages = advantages.unsqueeze(1)   # shape = [rollout_batch_size, 1]
        raw_rewards = raw_rewards.unsqueeze(1) # shape = [rollout_batch_size, 1]

        # tokenize rollouts
        tokenized_train_data = tokenize_prompt_and_output(prompt_strs=prompts, output_strs=responses, tokenizer=tokenizer)
        input_ids = tokenized_train_data["input_ids"] # shape = [rollout_batch_size, seq_len]
        labels = tokenized_train_data["labels"]       # shape = [rollout_batch_size, seq_len]
        response_mask = tokenized_train_data["response_mask"] # shape = [rollout_batch_size, seq_len]

        # 从old model中get per token log probs  for off-policy if using GRPO-clip
        # old_log_probs = None
        # if loss_type == "grpo_clip":
            # with torch.inference_mode():
                # input_ids = input_ids.to(train_device)
                # labels = labels.to(train_device)
                # log_probs = get_response_log_probs(model=old_model, input_ids=input_ids, labels=labels, return_token_entropy=True)
                # old_log_probs = log_probs["log_probs"].detach().cpu()

        # 从old model中get per token log probs  for off-policy if using GRPO-clip
        old_log_probs = None
        if loss_type == "grpo_clip":
            old_log_probs = []
            with torch.inference_mode():
                # microbatches inference
                n = rollout_batch_size // micro_train_batch_size
                for batch_idx in range(n):
                    start_idx = batch_idx * micro_train_batch_size
                    end_idx = (batch_idx + 1) * micro_train_batch_size

                    # get micro batch 
                    batch_input_ids = input_ids[start_idx:end_idx].to(train_device)
                    batch_labels = labels[start_idx:end_idx].to(train_device)
                    
                    # get policy log probs
                    with autocast_ctx:
                        log_probs = get_response_log_probs(model=model, input_ids=batch_input_ids, labels=batch_labels, return_token_entropy=False)
                        old_log_probs_tmp = log_probs["log_probs"].detach().cpu()
                        old_log_probs.append(old_log_probs_tmp)
            old_log_probs = torch.cat(old_log_probs, dim=0)
        
        # 训练模型
        synchronize()
        t0 = time.time()
        model.train()
        for epoch in range(epochs_per_rollout_batch):
            # shuffle data for this epoch
            perm = torch.randperm(rollout_batch_size)
            input_ids_epoch = input_ids[perm]
            labels_epoch = labels[perm]
            response_mask_epoch = response_mask[perm]
            raw_rewards_epoch = raw_rewards[perm]
            advantages_epoch = advantages[perm]
            old_log_probs_epoch = old_log_probs[perm] if old_log_probs is not None else None

            # microbatches training
            n = rollout_batch_size // micro_train_batch_size
            for batch_idx in range(n):
                start_idx = batch_idx * micro_train_batch_size
                end_idx = (batch_idx + 1) * micro_train_batch_size

                # get micro batch 
                batch_input_ids = input_ids_epoch[start_idx:end_idx].to(train_device)
                batch_labels = labels_epoch[start_idx:end_idx].to(train_device)
                batch_response_mask = response_mask_epoch[start_idx:end_idx].to(train_device)
                batch_raw_rewards = raw_rewards_epoch[start_idx:end_idx].to(train_device)
                batch_advantages = advantages_epoch[start_idx:end_idx].to(train_device)
                batch_old_log_probs = old_log_probs_epoch[start_idx:end_idx].to(train_device) if old_log_probs_epoch is not None else None
                
                # get policy log probs
                with autocast_ctx:
                    log_probs = get_response_log_probs(model=model, input_ids=batch_input_ids, labels=batch_labels, return_token_entropy=True)
                    log_probs_ = log_probs["log_probs"]   
                    entropy = log_probs["token_entropy"]
                    loss, metadata2 = grpo_microbatch_train_step(policy_log_probs=log_probs_,
                                                        response_mask=batch_response_mask,
                                                        gradient_accumulation_steps=gradient_accumulation_steps,
                                                        loss_type=loss_type,
                                                        raw_rewards=batch_raw_rewards if loss_type == "no_baseline" else None,
                                                        advantages=batch_advantages if loss_type != "no_baseline" else None,
                                                        old_log_probs = batch_old_log_probs,
                                                        cliprange=cliprange if loss_type == "grpo_clip" else None)
                    # loss, metadata2 = grpo_microbatch_train_step_masked_normalize(policy_log_probs=log_probs_, 
                    #                                     response_mask=batch_response_mask,
                    #                                     gradient_accumulation_steps=gradient_accumulation_steps,
                    #                                     loss_type=loss_type,
                    #                                     raw_rewards=batch_raw_rewards if loss_type == "no_baseline" else None,
                    #                                     advantages=batch_advantages if loss_type != "no_baseline" else None,
                    #                                     old_log_probs = batch_old_log_probs,
                    #                                     cliprange=cliprange if loss_type == "grpo_clip" else None)
                    train_loss = loss.detach() # for logging
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # gradient clipping
                    grad_clip_enabled = grad_clip > 0.0
                    if grad_clip_enabled:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
                    # step the optimizers
                    adamw_optimizer.step()
                    model.zero_grad(set_to_none=True)
                    synchronize()
                    t1 = time.time()
                    dt = t1 - t0
                    avg_entropy = masked_mean(entropy, batch_response_mask, dim=None).item()

                    # logging
                    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
                    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
                    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(trainstep + 1)) # debias the EMA
                    pct_done = 100 * trainstep / num_iterations 

                    if trainstep > 10:
                        total_training_time += dt # only count the time after the first 10 steps
                    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
                    print(f"step {trainstep:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} | dt: {dt * 1000:.2f}ms | total time: {total_training_time/60:.2f}m")
                    if trainstep % log_steps == 0:
                        log_data = {
                            "step": trainstep,
                            "total_training_time": total_training_time,
                            "train/loss": debiased_smooth_loss,
                            "train/dt": dt,
                            "train/token entropy": to_float_(avg_entropy),
                            "train/mean reward": to_float_(metadata["mean"]),
                        }
                        if grad_clip_enabled:
                            log_data["train/grad_norm"] = grad_norm
                        if loss_type == "grpo_clip":
                            clipped = metadata2["clipped"]
                            clip_fraction = clipped[batch_response_mask].float().mean()
                            log_data["train/clip_fraction"] = clip_fraction
                        wandb_run.log(log_data)
                    
                    trainstep+=1

        if grpo_step % eval_steps == 0:
            model.eval()
            load_policy_into_vllm_instance(model, vllm)
            
            overview = evaluate_vllm(vllm, 
                                    reward_fn = r1_zero_reward_fn, 
                                    prompts = [data["prompt"] for data in test_prompt],
                                    answers = [data["answer"] for data in test_prompt],
                                    eval_sampling_params = eval_sampling_params)
            accuracy = overview["correct"] / overview["count"]
            wandb_run.log({
                "eval/correct": overview["correct"],
                "eval/wrong answer": overview["answer_wrong"],
                "eval/wrong format": overview["format_wrong"],
                "eval/accuracy": accuracy,
                "eval_step": evalstep + 1
            })
            evalstep += 1
            model.save_pretrained(save_directory=output_path)
            tokenizer.save_pretrained(save_directory=output_path)
        

    print(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print(f"Total training time: {total_training_time/60:.2f}m")
    wandb_run.finish() # wandb run finish
    compute_cleanup()


def main(model_name_or_path: str, output_path: str, train_sample: int, loss_type: str):
    vllm = init_vllm(model_id=model_name_or_path, device=vllm_device, seed=seed, gpu_memory_utilization=0.6)
    model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                    ).to(device=train_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    train_grpo_experiment(model, tokenizer, vllm, output_path, train_sample, loss_type, 
                        n_grpo_steps=200, learning_rate=2e-5, advantage_eps=1e-6,
                        rollout_batch_size=256, group_size=8, sampling_temperature=1.0,
                        sampling_min_tokens=4, sampling_max_tokens=1024, 
                        epochs_per_rollout_batch=1, train_batch_size=256, gradient_accumulation_steps=256, # on-policy
                        use_std_normalization = True, cliprange=0.2)


def test_offpolicy(model_name_or_path: str, output_path: str, train_sample: int, loss_type: str):
    vllm = init_vllm(model_id=model_name_or_path, device=vllm_device, seed=seed, gpu_memory_utilization=0.6)
    model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                    ).to(device=train_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # # do a broad sweep
    # for epochs_per_rollout_batch in [2, 3, 4]:
    #     for train_batch_size in [32, 64, 128, 256]:
    #         train_grpo_experiment(model, tokenizer, vllm, output_path, train_sample, loss_type, 
    #                             n_grpo_steps=30, learning_rate=2e-5, advantage_eps=1e-6,
    #                             rollout_batch_size=256, group_size=8, sampling_temperature=1.0,
    #                             sampling_min_tokens=4, sampling_max_tokens=1024, 
    #                             epochs_per_rollout_batch=epochs_per_rollout_batch, train_batch_size=train_batch_size, gradient_accumulation_steps=train_batch_size,
    #                             use_std_normalization = True, cliprange=0.2)
    
    # # do a more focused sweep
    # for epochs_per_rollout_batch in [3]:
    #     for train_batch_size in [256]:
    #         train_grpo_experiment(model, tokenizer, vllm, output_path, train_sample, loss_type, 
    #                             n_grpo_steps=200, learning_rate=2e-5, advantage_eps=1e-6,
    #                             rollout_batch_size=256, group_size=8, sampling_temperature=1.0,
    #                             sampling_min_tokens=4, sampling_max_tokens=1024, 
    #                             epochs_per_rollout_batch=epochs_per_rollout_batch, train_batch_size=train_batch_size, gradient_accumulation_steps=train_batch_size,
    #                             use_std_normalization = True, cliprange=0.2)

    # 消融实验1：取消裁剪操作, 手动修改grpo.py文件中的compute_policy_gradient_loss函数
    epochs_per_rollout_batch = 3
    train_batch_size = 256
    train_grpo_experiment(model, tokenizer, vllm, output_path, train_sample, loss_type, 
                        n_grpo_steps=200, learning_rate=2e-5, advantage_eps=1e-6,
                        rollout_batch_size=256, group_size=8, sampling_temperature=1.0,
                        sampling_min_tokens=4, sampling_max_tokens=1024, 
                        epochs_per_rollout_batch=epochs_per_rollout_batch, train_batch_size=train_batch_size, gradient_accumulation_steps=train_batch_size,
                        use_std_normalization = True, cliprange=0.2)
    
    # # 消融实验2：使用简化的提示词, 将 r1_zero_reward_fn 替换成 question_only_reward_fn
    # epochs_per_rollout_batch = 2
    # train_batch_size = 256
    # train_grpo_experiment(model, tokenizer, vllm, output_path, train_sample, loss_type, 
    #                     n_grpo_steps=200, learning_rate=2e-5, advantage_eps=1e-6,
    #                     rollout_batch_size=256, group_size=8, sampling_temperature=1.0,
    #                     sampling_min_tokens=4, sampling_max_tokens=1024, 
    #                     epochs_per_rollout_batch=epochs_per_rollout_batch, train_batch_size=train_batch_size, gradient_accumulation_steps=train_batch_size,
    #                     use_std_normalization = True, cliprange=0.2)



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use", required=True
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    parser.add_argument("--train-samples", type=int, help="list of Number of traindata to use", default=128)
    parser.add_argument("--loss-type", type=str, help="采用的训练模式", default="reinforce_with_baseline")
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    # main(
    #     args.model_name_or_path,
    #     args.output_path,
    #     args.train_samples,
    #     args.loss_type
    # )
    test_offpolicy(
        args.model_name_or_path,
        args.output_path,
        args.train_samples,
        args.loss_type
    )
    logger.info("finished running %s", sys.argv[0])