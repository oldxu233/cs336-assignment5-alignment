"""
Use Qwen2.5-Math-1.5B test zero-shot performance.

Running:

```
CUDA_VISIBLE_DEVICES=0 python cs336_alignment/math_baseline.py \
    --input-path "data/gsm8k/test.jsonl" \
    --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B" \
    --num-gpus 1 \
    --output-path "gsm8k_QwenMath_test_out.json"

CUDA_VISIBLE_DEVICES=0 python cs336_alignment/math_baseline.py \
    --input-path "data/gsm8k/train.jsonl" \
    --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B" \
    --num-gpus 1 \
    --output-path "gsm8k_QwenMath_train_out.json"

CUDA_VISIBLE_DEVICES=0,2 python cs336_alignment/math_baseline.py \
    --input-path "data/gsm8k/test.jsonl" \
    --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen3-32B-AWls" \
    --num-gpus 2 \
    --output-path "gsm8k_Qwen32B_test_out.json"

CUDA_VISIBLE_DEVICES=0,2 python cs336_alignment/math_baseline.py \
    --input-path "data/gsm8k/train.jsonl" \
    --model-name-or-path "/home/xqzzz1/codes/myvllm/model/Qwen3-32B-AWls" \
    --num-gpus 2 \
    --output-path "gsm8k_Qwen32B_train_out.json"
```
"""
from vllm import LLM, SamplingParams
from xopen import xopen
from typing import Callable, List, Dict, Any
from drgrpo_grader import r1_zero_reward_fn, new_r1_zero_reward_fn

import argparse
import json
import logging
import sys
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)
# QwenMath = "/home/xqzzz1/codes/myvllm/model/Qwen2.5-Math-1.5B"
# Llama_8B = "/home/xqzzz1/codes/myvllm/model/Llama-3.1-8B"
prompt_path = "cs336_alignment/prompts/r1_zero.prompt"


def load_dataset_and_format_prompts(input_dataset_path: str, prompt_template: str):
    # 使用 r1_zero 提示模板将样本格式化为语言模型可接受的字符串提示；
    with open(prompt_template, "r") as f:
        r1_zero_prompt = f.read()
    
    prompts = []
    answers = []
    with xopen(input_dataset_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(r1_zero_prompt.format(question=data["question"]))
            answers.append(data["answer"])
    logger.info(f"Read {len(answers)} model responses from {input_dataset_path}")
    return prompts, answers


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

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """
    对一组提示评估语言模型，计算评估指标，并将结果序列化到磁盘。
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    evaluate_results = []
    for prompt, generated_text, answer in tqdm(zip(prompts, generated_texts, answers), total=len(prompts), desc="Evaluating"):
        extracted_answer = extract_answer(answer)
        reward_dict = reward_fn(generated_text, extracted_answer)
        
        evaluate_results.append({
            "prompt": prompt,
            "response": generated_text,
            "answer": answer,
            "extracted_answer": extracted_answer,
            "reward": reward_dict["reward"],
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
        })
    return evaluate_results

def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    output_data = {"metrics": metrics, "results": results}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved {len(results)} results to {output_path}")

def calculate_metrics(results: List[Dict[str, Any]]):
    """Calculate and log evaluation metrics."""
    total_examples = len(results)
    correct_both = sum(1 for r in results if r["format_reward"] == 1 and r["answer_reward"] == 1)
    correct_format_wrong_answer = sum(1 for r in results if r["format_reward"] == 1 and r["answer_reward"] == 0)
    wrong_format = sum(1 for r in results if r["format_reward"] == 0)

    avg_total_reward = sum(r["reward"] for r in results) / total_examples
    avg_format_reward = sum(r["format_reward"] for r in results) / total_examples
    avg_answer_reward = sum(r["answer_reward"] for r in results) / total_examples

    metrics = {
        "total_examples": total_examples,
        "correct_both": correct_both,
        "correct_format_wrong_answer": correct_format_wrong_answer,
        "wrong_format": wrong_format,
        "avg_total_reward": avg_total_reward,
        "avg_format_reward": avg_format_reward,
        "avg_answer_reward": avg_answer_reward,
        "accuracy": avg_answer_reward
    }
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total examples: {total_examples}")
    print(f"\nCategory breakdown:")
    print(f"  (1) Correct format + Correct answer: {correct_both} ({100*correct_both/total_examples:.1f}%)")
    print(f"  (2) Correct format + Wrong answer: {correct_format_wrong_answer} ({100*correct_format_wrong_answer/total_examples:.1f}%)")
    print(f"  (3) Wrong format: {wrong_format} ({100*wrong_format/total_examples:.1f}%)")
    print(f"\nAverage rewards:")
    print(f"  Total reward: {avg_total_reward:.4f}")
    print(f"  Format reward: {avg_format_reward:.4f}")
    print(f"  Answer reward (accuracy): {avg_answer_reward:.4f}")
    print("="*60 + "\n")
    return metrics


def bulid_llm_and_params(model_name_or_path, num_gpus):
    # 为每个样本生成模型输出；创建采样参数对象，设置在换行符处停止生成
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output = True
    )
    # # Qwen32B
    # llm = LLM(model=model_name_or_path, tensor_parallel_size=num_gpus,
    #           gpu_memory_utilization=0.80, max_model_len=4096)
    
    # Qwen Math
    llm = LLM(model=model_name_or_path, tensor_parallel_size=num_gpus)
    return llm, sampling_params
    

def main(input_path, model_name_or_path, num_gpus, output_path):
    prompts, answers = load_dataset_and_format_prompts(input_path, prompt_path)
    llm, sampling_params = bulid_llm_and_params(model_name_or_path, num_gpus)
    
    # 针对qwen2.5 math模型
    eval_results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)   
    # # 针对qwen32B 模型
    # eval_results = evaluate_vllm(llm, new_r1_zero_reward_fn, prompts, answers, sampling_params) 

    # 将样本、模型生成结果及对应的评估分数序列化保存到磁盘，供后续问题分析使用。
    metrics = calculate_metrics(eval_results)
    save_results(eval_results, metrics, output_path)



if __name__ == "__main__":
    # prompts, answers = load_dataset_and_format_prompts("data/test.jsonl", prompt_path)
    # print(f"prompts: {prompts}, answers: {answers}")


    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to file with model predictions (JSONL format with key 'output')",
    )
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use", required=True
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model_name_or_path,
        args.num_gpus,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
    