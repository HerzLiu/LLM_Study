---
tags: [Happy-LLM, 第4章, LLM, SFT, 指令微调]
chapter: 4
section: 4.2.2
---

# 4.2.2 SFT（有监督微调）—— 第二阶段：教他听人话

⬅ [05-Pretrain预训练](05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)　|	➡ [07-RLHF人类反馈强化学习](07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)

> 🎯 **目标**：把"博览群书的书呆子"变成"会按指令办事的助理"
> 💰 **成本**：数十 GPU × 数天（远低于 [Pretrain](05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)）

---

## 🎬 故事比喻：从"会背书"到"听话办事"

[Pretrain](05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md) 出来的 LLM 像个**博览群书但不求甚解的书生**：
- 你说"今天天气"，他会接"很好" ← 只会续写
- 你问"今天天气怎么样？"，他可能莫名其妙地写一堆类似的问题
- 因为他只学过 [CLM](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-GPT.md#-%E9%A2%84%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1clm%E5%9B%A0%E6%9E%9C%E8%AF%AD%E8%A8%80%E5%BB%BA%E6%A8%A1)，只会"预测下一个 token"

→ **SFT 就是教这个书生：你说的话是"指令"，要按指令"办事"**。

---

## 🔧 SFT vs 传统微调

### 传统 PLM 的微调（任务专用）

[BERT](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02-BERT.md) 时代：
- 想做情感分类 → 单独微调一个情感分类模型
- 想做实体识别 → 单独微调一个实体识别模型
- **一个任务一个模型**

### LLM 的 SFT（指令微调）

不再针对特定下游任务，而是**训练通用指令遵循能力**：
- 一份训练集涵盖**几十种任务**（翻译、摘要、问答、写代码…）
- 让模型学会"看到指令就能办事"
- **一个模型搞定所有任务**

---

## 📦 SFT 数据格式

### 三件套（Alpaca 风格）

最经典的 SFT 数据格式（Stanford Alpaca）：

```json
{
    "instruction": "用户的指令（要做什么）",
    "input":       "执行指令需要的输入（可选，没有则置空）",
    "output":      "模型应该给出的回复"
}
```

### 例子

```json
{
    "instruction": "将下列文本翻译成英文：",
    "input":       "今天天气真好",
    "output":      "Today is a nice day!"
}
```

### LLaMA 的 SFT 模板

```
### Instruction:
{将 instruction 和 input 拼接}

### Response:
{output}
```

具体例子（输入给模型）：
```
### Instruction:
将下列文本翻译成英文：今天天气真好

### Response:
Today is a nice day!
```

---

## 🎯 SFT 训练原理：仍然是 CLM

> **SFT ≠ 全新训练任务，它本质上还是 CLM！**

### 关键点

模型预测的目标是：`input + output` 的拼接
但**只对 output 部分计算 loss**（input 部分 mask 掉）

### 直观理解

| Token 位置                                                 | 内容   | 是否算 loss |
| -------------------------------------------------------- | ---- | -------- |
| `### Instruction:\n将下列文本翻译成英文：今天天气真好\n\n### Response:\n` | 输入指令 | ❌ 不算     |
| `Today is a nice day!`                                   | 期望回复 | ✅ 算      |

→ 模型仍然是 token-by-token 地"预测下一个"，但**只在"回复部分"惩罚错误**。
→ 这样模型既保留了语言生成能力，又学会了"对指令做回复"。

---

## 📊 SFT 数据规模和配比

### 数据量

| 场景 | 推荐样本量 |
|---|---|
| 单任务微调 | 500~1000 |
| **泛化指令遵循（主流 LLM）** | **几 B token**（数百万~千万样本） |
 
### 任务配比（InstructGPT 例子）

OpenAI 训练 ChatGPT 前身 InstructGPT 时的指令分布：

| 任务类型 | 占比 |
|---|---|
| 文本生成 | **45.6%** |
| 开放域问答 | 12.4% |
| 头脑风暴 | 11.2% |
| 聊天 | 8.4% |
| 文本转写 | 6.6% |
| 文本总结 | 4.2% |
| 文本分类 | 3.5% |
| 其他 | 3.5% |
| 特定域问答 | 2.6% |
| 文本抽取 | 1.9% |

→ 配比平衡，覆盖多种用户需求。

---

## 💰 SFT 数据从哪来？

### 方案 1：人工标注（质量最高，成本最贵）

- ChatGPT 成功的核心：**OpenAI 雇了大量高质量标注员**
- 但成本极高，企业一般不开源

### 方案 2：用更强的 LLM 生成（Self-Instruct）

经典开源数据集 **Alpaca** 的做法：

```
1. 准备一些"种子 Prompt"
2. 用 ChatGPT/GPT-4 基于种子生成更多 Prompt
3. 用 ChatGPT/GPT-4 给生成的 Prompt 写回复
4. 得到一批 (instruction, input, output) 训练数据
```

优势：成本低、生成快
劣势：质量受限于教师模型、可能有版权问题

### 方案 3：开源数据集

| 数据集               | 来源                         | 特点       |
| ----------------- | -------------------------- | -------- |
| **Alpaca**        | Self-Instruct from GPT-3.5 | 52K 英文样本 |
| **ShareGPT**      | 用户分享的 ChatGPT 对话           | 多轮对话     |
| **Belle**         | 中文版 Alpaca                 | 中文 SFT   |
| **MOSS**          | 复旦开源                       | 中文多轮     |
| **OpenAssistant** | 众包标注                       | 多语言      |

---

## 💬 多轮对话怎么训？

### 单轮 vs 多轮

| 类型 | 例子 |
|---|---|
| **单轮** | 一问一答 |
| **多轮** | 模型能记住对话历史 |

### 没多轮能力的样子

```
用户：你好，我是 Datawhale 成员。
模型：您好，请问有什么可以帮助您？

用户：你知道 Datawhale 是什么吗？
模型：不好意思，我不知道。  ← 居然忘了用户刚说的！
```

### 有多轮能力的样子

```
用户：你好，我是 Datawhale 成员。
模型：您好，请问有什么可以帮助您？

用户：你知道 Datawhale 是什么吗？
模型：Datawhale 是一个开源组织。  ← 利用了历史信息
```

### 多轮对话**只来自 SFT**

→ **预训练不会教多轮对话**，必须 SFT 阶段构造多轮数据训练。

### 构造多轮对话样本的 3 种方式

假设原始对话是：
```
<prompt_1><completion_1>
<prompt_2><completion_2>
<prompt_3><completion_3>
```

#### 方式 1：只拟合最后一轮回复（❌ 信息浪费）

```
input  = <prompt_1><completion_1><prompt_2><completion_2><prompt_3>
output = <completion_3>
```
**问题**：浪费了 completion_1、completion_2 的训练信号。

#### 方式 2：拆成 N 个独立样本（❌ 计算浪费）

```
样本 1: <prompt_1> → <completion_1>
样本 2: <prompt_1><completion_1><prompt_2> → <completion_2>
样本 3: <prompt_1><completion_1><prompt_2><completion_2><prompt_3> → <completion_3>
```
**问题**：前缀重复计算了 N 次，浪费算力。

#### ✅ 方式 3：一次性预测每一轮（最优）

```
input  = <prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
output = [MASK]   <completion_1>[MASK]   <completion_2>[MASK]   <completion_3>
```

只对每个 `completion_i` 算 loss，其他 mask 掉。
**优势**：
- 没有信息浪费
- 没有计算浪费
- 一次 forward 训完整个对话

#### ⭐ 为什么方式 3 可行？

因为 [掩码自注意力](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/04-%E6%8E%A9%E7%A0%81%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B.md) 的特性：
- 每个位置**只能看到前面**
- 所以 completion_2 的预测**只依赖 prompt_1, completion_1, prompt_2**
- 后面的 completion_3 **不会"污染"** completion_2 的预测
- → 一次性算多轮 loss 完全等价于分轮算

→ **目前所有 LLM 都用方式 3**。

---

## 🎯 SFT 的关键设计原则

### 1. 数据质量 > 数据数量

- **几千条精挑细选 > 几百万条粗制滥造**
- Meta 的 LIMA 论文证明：**1000 条高质量样本**就能让 LLaMA 表现不错

### 2. 任务多样性 > 单任务深度

- 覆盖各种任务类型（生成、问答、推理、代码…）
- 让模型获得**泛化的指令遵循能力**

### 3. 格式一致

- 训练和推理用同一套模板
- 一旦改格式，模型可能"水土不服"

### 4. 长度多样

- 既有短回复（一句话）
- 也有长回复（几百字）
- 教会模型"按需输出"

---

## ⚠️ 小白避坑

1. **SFT 不能"复活"预训练没学过的知识**
   - 模型不知道的事，SFT 也教不会
   - SFT 只是"激活"和"格式化"
2. **SFT 数据格式必须和推理一致**
   - 训练用 `### Instruction:` 模板，推理也得用
   - 否则模型不认
3. **多轮对话数据是 SFT 的关键**
   - 想让模型支持对话，必须有多轮训练数据
4. **不要过度微调**
   - SFT 太久会让模型"过拟合"指令格式
   - 失去通用语言能力（**灾难性遗忘**）
5. **小心 SFT 数据的版权和合规**
   - 用 GPT-4 生成的数据可能违反 OpenAI 服务条款

---

## 📌 SFT 完成 = "Chat Model"

SFT 完成后的模型称为 **Chat / Instruct 模型**：
- LLaMA-2-Base → **LLaMA-2-Chat**
- Qwen-Base → **Qwen-Chat**
- 直接可对话使用

→ 但**这还不够好**，需要 RLHF 进一步对齐。

---

## 🔗 延伸阅读

- 上一节：[05-Pretrain预训练](05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)
- 下一节：[07-RLHF人类反馈强化学习](07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md) —— 第三阶段
- 实战：[第 5 章 SFT 实战](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/10-SFT%E8%AE%AD%E7%BB%83.md)
- 高效微调：[第 6 章 LoRA](../06-%E7%AC%AC6%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%AE%9E%E8%B7%B5/05-LoRA%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)

---

⬅ [05-Pretrain预训练](05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)　|	➡ [07-RLHF人类反馈强化学习](07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)
