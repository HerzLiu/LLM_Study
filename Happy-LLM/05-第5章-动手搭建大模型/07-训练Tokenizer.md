---
tags: [Happy-LLM, 第5章, 实战, Tokenizer, BPE]
chapter: 5
section: 5.2
---

# 5.2 训练 Tokenizer

⬅ [06-完整LLaMA2模型](06-%E5%AE%8C%E6%95%B4LLaMA2%E6%A8%A1%E5%9E%8B.md)	|	➡ [08-数据集构造](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md)

> 🎯 上一节我们造好了模型"大脑"，这一节给它配一张**翻译卡**：把汉字 → 数字 id。

---

## 🎬 故事比喻：翻译卡

模型只懂数字 0~6143（词表大小），不懂汉字。
**Tokenizer 就是一张翻译卡**：
- "今天天气真好" → `[1234, 5678, 9012, 3456]`
- `[1234, 5678, 9012, 3456]` → "今天天气真好"

→ 模型 input 都是 token id，所以 Tokenizer 是必备的"前处理"组件。

---

## 📊 Tokenizer 三大流派

| 类型 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| **Word-based** | 按空格/标点切单词 | 直观、符合直觉 | OOV 问题严重，中文没空格 |
| **Character-based** | 每个字符一个 token | 灵活、无 OOV | 序列超长、计算重 |
| **Subword** ⭐ | 介于词和字符之间 | **平衡**，处理罕见词友好 | 实现稍复杂 |

→ 现代 LLM **全部用 Subword**。

### Subword 三大算法

| 算法 | 代表用户 | 思路 |
|---|---|---|
| **BPE**（Byte Pair Encoding） | GPT、LLaMA | 反复合并**最高频**字符对 |
| **WordPiece** | BERT | 反复合并**最大化似然**的字符对 |
| **Unigram** | T5 | 概率模型分割 |

本节我们用 **BPE**（最常用、最简单）。

---

## 🔧 BPE 工作原理（速览）

```
初始词表：单个字符
  ["h", "e", "l", "o"]

输入文本："hello hello hello"

Step 1: 统计字符对频率
  "he": 3,  "el": 3,  "ll": 3,  "lo": 3, ...

Step 2: 合并最高频的对
  ["h", "e", "l", "o", "he"]   ← "h" + "e" 合并为 "he"

Step 3: 重新统计 → 合并 → 重复 N 次
  ["h", "e", "l", "o", "he", "el", "lo", "hel", "hello"]

最终：能用 "hello" 这一个 token 表示
```

→ **越频繁的组合 → 越早被合并 → 越短的编码**。

---

## 🐍 完整训练代码

```python
import os, json, random
from typing import Generator
from transformers import AutoTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
from tokenizers.normalizers import NFKC

random.seed(42)

# ===== Step 1: 读 JSONL 数据 =====
def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """流式读 JSONL，节省内存"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                continue
            except KeyError as e:
                print(e)
                continue


# ===== Step 2: 训练 BPE Tokenizer =====
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 6144) -> None:
    """训练并保存 BPE Tokenizer"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 初始化 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2. 文本规范化（NFKC：统一全角/半角、Unicode 规范化）
    tokenizer.normalizer = NFKC()

    # 3. 预分词：字节级（兼容所有 UTF-8 字符）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 4. 解码器：与预分词器配套
    tokenizer.decoder = decoders.ByteLevel()

    # 5. 特殊 token（ChatML 风格）
    special_tokens = [
        "<unk>",          # 未知 token
        "<s>",            # 句子开始（不强制用）
        "</s>",           # 句子结束（不强制用）
        "<|im_start|>",   # 对话角色开始
        "<|im_end|>",     # 对话角色结束
    ]

    # 6. 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,                          # 频次 < 2 的不要
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 7. 开始训练（流式读，省内存）
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # 8. 验证特殊 token ID
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2
    assert tokenizer.token_to_id("<|im_start|>") == 3
    assert tokenizer.token_to_id("<|im_end|>") == 4

    # 9. 保存
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")


# ===== Step 3: 创建配置文件（给 Hugging Face 用）=====
def create_tokenizer_config(save_dir: str) -> None:
    """创建 tokenizer_config.json 和 special_tokens_map.json"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        # 对话模板（与 Qwen2.5 一致）
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)


# ===== Step 4: 评估 Tokenizer =====
def eval_tokenizer(tokenizer_path: str) -> None:
    """测试 Tokenizer 的基本功能"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("\n=== Tokenizer 基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个 AI 助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\n=== 聊天模板 ===")
    print(prompt)

    # 测试编码解码
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("\nDecoded matches original:", decoded == prompt)


# ===== 主入口 =====
def main():
    train_tokenizer(
        data_path="your_data.jsonl",
        save_dir="tokenizer_k",
        vocab_size=6144,
    )
    eval_tokenizer("tokenizer_k")


if __name__ == '__main__':
    main()
```

---

## 📝 代码深度解读

### 关键 1：5 个特殊 token 的含义

```python
special_tokens = [
    "<unk>",          # ID=0 未知词
    "<s>",            # ID=1 序列开始（备用）
    "</s>",           # ID=2 序列结束（备用）
    "<|im_start|>",   # ID=3 对话角色开始
    "<|im_end|>",     # ID=4 对话角色结束
]
```

- `<|im_start|>`/`<|im_end|>` 是 **ChatML** 格式，Qwen、GPT-4 都用这种风格
- **ID 必须固定**（assert 检查），因为后续 [SFT](10-SFT%E8%AE%AD%E7%BB%83.md) 训练要按 ID 匹配 token 序列

### 关键 2：ByteLevel 预分词

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

- **ByteLevel**：把文本拆成字节级别（UTF-8 字节）
- 优点：**支持任意 Unicode 字符**（汉字、emoji、特殊符号都行）
- LLaMA、GPT、Qwen 都用 ByteLevel

### 关键 3：NFKC 规范化

```python
tokenizer.normalizer = NFKC()
```

- **NFKC**：Unicode 规范化（如全角"Ａ" → 半角"A"）
- 让"看起来一样"的字符**有相同的编码**

### 关键 4：流式训练

```python
texts = read_texts_from_jsonl(data_path)  # generator
tokenizer.train_from_iterator(texts, trainer=trainer, length=...)
```

- `read_texts_from_jsonl` 是 **生成器**（不一次性加载到内存）
- 几 GB 的数据也能训
- 没这个的话**几十 GB 数据会爆内存**

### 关键 5：Chat Template

```python
"chat_template": (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    ...
)
```

这是 **Jinja 模板**，由 `tokenizer.apply_chat_template()` 调用：

输入：
```python
[
    {"role": "system", "content": "你是 AI"},
    {"role": "user", "content": "你好"},
]
```

输出：
```
<|im_start|>system
你是 AI<|im_end|>
<|im_start|>user
你好<|im_end|>
```

→ **统一对话格式**，让训练和推理都用同一套规则。

---

## 🔬 训练后效果

```
=== Tokenizer 基本信息 ===
Vocab size: 6144
Special tokens: ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']
Special token IDs: [3, 4, 0, 1, 2]

=== 聊天模板 ===
<|im_start|>system
你是一个 AI 助手。<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you.<|im_end|>
```

✅ Tokenizer 训练完成！

---

## 📦 词表大小怎么选？

| 词表大小 | 适用场景 |
|---|---|
| 5K~10K | 单语种小模型（本章用 6144） |
| 30K~50K | GPT-2、BERT 风格 |
| **100K~150K** | **现代 LLM 主流**（LLaMA-3、Qwen-2） |

**权衡**：
- 词表大 → 每个 token 表达更多内容 → 序列更短 → 推理更快
- 词表大 → Embedding 层参数更多 → 模型更重
- 词表大 → 训练数据要求更多

---

## ⚠️ 小白避坑

1. **特殊 token ID 必须固定**
   - 用 assert 严格检查
   - 否则 [SFT 的 loss_mask](10-SFT%E8%AE%AD%E7%BB%83.md) 会错位
2. **`min_frequency` 别设太大**
   - 太大词表里只有高频词，小语料就训不动
3. **ByteLevel 解码可能加空格**
   - 这是 ByteLevel 的"特性"（处理英文空格用）
   - 中文场景偶尔会有"<|im_start|> user"这种意外空格
   - 不影响功能但**测试时要注意**
4. **训练数据**
   - 用预训练同源数据训 tokenizer（如本章用"出门问问"语料训完后给预训练用）
   - 数据**分布要匹配**，否则 OOV 多

---

## 🔗 延伸阅读

- 上一节：[06-完整LLaMA2模型](06-%E5%AE%8C%E6%95%B4LLaMA2%E6%A8%A1%E5%9E%8B.md)
- 下一节：[08-数据集构造](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md)
- 理论篇：[子词切分](../01-%E7%AC%AC1%E7%AB%A0-NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/03-NLP%E4%B9%9D%E5%A4%A7%E4%BB%BB%E5%8A%A1.md#%E2%91%A1-%E5%AD%90%E8%AF%8D%E5%88%87%E5%88%86subword--%E9%87%8D%E8%A6%81)

---

⬅ [06-完整LLaMA2模型](06-%E5%AE%8C%E6%95%B4LLaMA2%E6%A8%A1%E5%9E%8B.md)	|	➡ [08-数据集构造](08-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E9%80%A0.md)
