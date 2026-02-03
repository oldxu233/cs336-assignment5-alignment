# 解析下数据集 openai/gsm8k
# export HF_ENDPOINT=https://hf-mirror.com 临时设置镜像

from datasets import load_dataset, config

ds = load_dataset("json",
                  data_files={
                      "train": "data/gsm8k/train.jsonl",
                      "test": "data/gsm8k/test.jsonl"
                  })
print("缓存根目录:", config.HF_CACHE_HOME)
print("GSM8K 数据集缓存路径:", ds['train'].cache_files)

print("=== 数据集信息 ===")
print(f"数据集类型: {type(ds)}")
print(f"数据集键: {list(ds.keys())}")
print(f"训练集大小: {len(ds['train'])} 条")
print(f"测试集大小: {len(ds['test'])} 条")

print("\n=== 第一条训练样本 ===")
print(ds['train'][0])

print("\n=== 数据结构 ===")
print(ds['train'].features)

print("\n=== 缓存确认 ===")
print("训练集缓存文件:", ds['train'].cache_files)
print("测试集缓存文件:", ds['test'].cache_files)

# 统计一下
if 'question' in ds['train'].features:
    print(f"\n第一个问题: {ds['train'][0]['question'][:100]}...")
if 'answer' in ds['train'].features:
    print(f"答案长度: {len(ds['train'][0]['answer'])} 字符")