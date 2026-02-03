"""
# 安装项目但不安装依赖
pip install -e . --no-deps

CUDA_VISIBLE_DEVICES=0,2 python scripts/sft.py --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B" --output-path "sft_logs/" --data-type "correct" --train-samples 128,256
CUDA_VISIBLE_DEVICES=0,2 python scripts/sft.py --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B" --output-path "sft_logs/" --data-type "raw" --train-samples 128,256,512
"""

import torch
import wandb 
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from unittest.mock import patch
import random
import json
from typing import Callable, List, Dict, Any
import argparse
import logging
import sys
import re
import time
import os

from cs336_basics.nanochat.common import DummyWandb, compute_cleanup
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)
prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
train_data_path = "sft_logs/gsm8k_QwenMath_train_out.json"
# train_data_path = "sft_logs/gsm8k_QwenMath_train_correct_out.json"
test_data_path = "data/gsm8k/test.jsonl"
seed = 42
train_device = 'cuda:0'
vllm_device = "cuda:1"

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
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
        qa.append({"prompt": item["prompt"], "response": item["response"]})
    return qa


def load_dataset_and_format_qa(train_sample):
    """
    args: 
        train_sample: int, 使用的训练样本数量
    returns: 
        train_data: list[dict], 格式化后的训练数据
        test_data: list[dict], 格式化后的测试数据
    
    例如：train_sample = 100, 使用到前100个训练样本
    输入：
    train_data = 
    {
        "metrics": {
            ...
        },
        "results": [
            {
            "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAssistant: <think>",
            "response": " We need to calculate the total number of eggs laid per day, subtract the eggs eaten by Janet and the eggs baked for her friends, then multiply the remaining eggs by the price per egg. </think> <answer> Let's break it down step by step. First, calculate the total eggs laid per day: 16 eggs. Next, subtract the eggs eaten and the eggs baked for her friends: 16 - 3 - 4 = 9 eggs. Finally, multiply the remaining eggs by the price per egg: 9 eggs * $2/egg = $18. So, Janet makes $18 every day at the farmers' market.\n</think>",
            "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18",
            "extracted_answer": "18",
            "reward": 0.0,
            "format_reward": 0.0,
            "answer_reward": 0.0
            },...]
    }
    输出：
    train_data = list[{"prompt": "A conversation between User and Assistant....", "response": " We need to calculate the total numb..."}, ..., {}] 0-99
    """
    train_data = load_train_dataset(train_data_path) 
    # train_data = train_data[:train_sample]
    train_data = random.sample(train_data, min(len(train_data), train_sample))
    test_data = load_test_dataset(test_data_path)           # list[{"question": "1+1=?", "answer": "2"}, ..., {}]
    test_data = format_test_dataset(test_data, prompt_path) # list[{"prompt": "A conversation between User and Assistant... 1+1=?", "answer": "2"}, ...,{}]
    return train_data, test_data

     
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

def get_batch(tokenized_train_data: dict[str, torch.Tensor], batch_size: int, device: str):
    batch_indices = random.sample(range(len(tokenized_train_data["input_ids"])), batch_size)
    return {k: v[batch_indices].to(device) for k, v in tokenized_train_data.items()}



def train_sft_experiment(model, tokenizer, vllm, output_path: str, train_sample: int, data_type: str):
    # 允许通过配置器在命令行界面（CLI）中覆盖设置
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    user_config = {k: globals()[k] for k in config_keys} # will be useful for logging

    # Compute init
    autocast_ctx = torch.amp.autocast(device_type=train_device, dtype=torch.bfloat16)
    synchronize = torch.cuda.synchronize
    get_max_memory = torch.cuda.max_memory_allocated

    # wandb logging init
    run = exp_name = f"sft_gsm8k_{train_sample}_{data_type}_on_QwenMath_" + time.strftime('%Y%m%d_%H%M%S') # wandb 运行名称（"dummy" 表示不使用 wandb 日志）
    use_dummy_wandb = run == "dummy"
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="cs336-sft", name=run, config=user_config)
    
    # train
    batch_size = 1
    grad_accum_steps = 64
    num_iterations = 256
    grad_clip = 1.0               # 梯度裁剪阈值（L2 范数上限）
    
    # eval
    log_steps = 16
    eval_steps: int = 16

    adamw_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    train_qa, test_prompt = load_dataset_and_format_qa(train_sample)
    tokenized_train_data = tokenize_prompt_and_output(prompt_strs=[data["prompt"] for data in train_qa], 
                                                      output_strs=[data["response"] for data in train_qa], 
                                                      tokenizer=tokenizer)
    
    
    step = 0
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    while True:
        # # 检查训练模型的设备
        # model_params = list(model.parameters())
        # if model_params:
        #     model_device = model_params[0].device
        #     print(f"训练模型实际设备: {model_device}")
        # cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
        # print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        last_step = step == num_iterations
        if last_step:
            break
        # train_batch = get_batch(tokenized_train_data, batch_size=batch_size, device=train_device)
        # input_ids = train_batch["input_ids"]
        # labels = train_batch["labels"]
        # response_mask = train_batch["response_mask"]

        synchronize()
        t0 = time.time()
        model.train()
        for micro_step in range(grad_accum_steps):
            train_batch = get_batch(tokenized_train_data, batch_size=batch_size, device=train_device)
            input_ids = train_batch["input_ids"]
            labels = train_batch["labels"]
            response_mask = train_batch["response_mask"]

            with autocast_ctx:
                log_probs = get_response_log_probs(model=model, input_ids=input_ids, labels=labels, return_token_entropy=True)
                log_probs_ = log_probs["log_probs"]
                entropy = log_probs["token_entropy"]
                loss, _ = sft_microbatch_train_step(policy_log_probs=log_probs_, response_mask=response_mask, gradient_accumulation_steps=grad_accum_steps)
                train_loss = loss.detach() # for logging
        
        # gradient clipping
        grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
        if grad_clip_enabled:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
        # step the optimizers
        adamw_optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        t1 = time.time()
        dt = t1 - t0

        # logging
        ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
        pct_done = 100 * step / num_iterations

        if step > 10:
            total_training_time += dt # only count the time after the first 10 steps
        print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
        print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} | dt: {dt * 1000:.2f}ms | total time: {total_training_time/60:.2f}m")
        if step % log_steps == 0:
            log_data = {
                "step": step,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/dt": dt,
                "train/entropy": to_float_(entropy.mean()),
                "train/response entropy": to_float_(entropy[response_mask].mean()),
                "train/prompt entropy": to_float_(entropy[~response_mask].mean()),
            }
            if grad_clip_enabled:
                log_data["train/grad_norm"] = grad_norm
            wandb_run.log(log_data)

        if step % eval_steps == 0:
            load_policy_into_vllm_instance(model, vllm)
            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=1024,
                stop=["</answer>"],
                include_stop_str_in_output = True)
            overview = evaluate_vllm(vllm, 
                                     reward_fn = r1_zero_reward_fn, 
                                     prompts = [data["prompt"] for data in test_prompt],
                                     answers = [data["answer"] for data in test_prompt],
                                     eval_sampling_params = sampling_params)
            accuracy = overview["correct"] / overview["count"]
            wandb_run.log({
                "eval/correct": overview["correct"],
                "eval/wrong answer": overview["answer_wrong"],
                "eval/wrong format": overview["format_wrong"],
                "eval/accuracy": accuracy,
                "eval_step": step + 1
            })

            model.save_pretrained(save_directory=output_path)
            tokenizer.save_pretrained(save_directory=output_path)
        # state update
        step += 1

    print(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print(f"Total training time: {total_training_time/60:.2f}m")
    wandb_run.finish() # wandb run finish
    compute_cleanup()


def main(model_name_or_path: str, output_path: str, train_samples: list[int], data_type: str):
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    vllm = init_vllm(model_id=model_name_or_path, device=vllm_device, seed=seed, gpu_memory_utilization=0.6)
        
    for train_sample in train_samples:
        model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                    ).to(device=train_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        train_sft_experiment(model, tokenizer, vllm, f"{output_path}_N{train_sample}_{data_type}", train_sample, data_type)
    

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
    parser.add_argument("--train-samples", help="list of Number of traindata to use", type=lambda s: [int(item) for item in s.split(',')], default=128)
    parser.add_argument(
        "--data-type",
        type=str,
        help="使用的训练数据是否经过筛选? correct表示经过了筛选 只保留reward=1的数据集合",
        default="correct"
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.model_name_or_path,
        args.output_path,
        args.train_samples,
        args.data_type
    )
    logger.info("finished running %s", sys.argv[0])
