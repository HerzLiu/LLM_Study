---
tags: [Hello-Agents, 第3章, 提示工程, 分词, 采样参数]
chapter: 3
section: 3.2
---

# 3.2 与 LLM 交互(Agent 必精读)

⬅ [01-LLM架构速览](01-LLM%E6%9E%B6%E6%9E%84%E9%80%9F%E8%A7%88.md)	|	➡ [03-缩放法则与局限](03-%E7%BC%A9%E6%94%BE%E6%B3%95%E5%88%99%E4%B8%8E%E5%B1%80%E9%99%90.md)

> ⚠️ **本节 Agent 强相关,精读**。
> 提示工程 / 采样参数 / 分词 —— **每一项都直接影响你的 Agent 性能和成本**。

---

## 🎬 故事比喻:跟 LLM "聊得溜"

LLM 就像一个**超级聪明但傲娇的实习生**:
- **说不清要求** → 它瞎做
- **温度太高** → 它瞎写
- **不知道 token 怎么算** → 你**钱包流血**

→ **3 件事必须会**:写 prompt + 调采样参数 + 算 token 成本。

---

## 🎛 3.2.1 模型采样参数(三大旋钮)

> Agent 的 LLM 调用里,**温度 / Top-k / Top-p** 是必调参数。

### 背景:LLM 怎么"选下一个词"?

每一步 LLM 都会输出**一个概率分布**(词表里每个词的概率)。
**怎么从这个分布选?** —— 看采样参数。

### 旋钮 1:Temperature(温度)

公式:`softmax(logits / T)`

| T 范围 | 效果 | 适用 |
|---|---|---|
| **0.0** | 完全确定(贪心,总选最高概率) | Agent 工具调用、数学、代码 |
| **0.1~0.3** | 精准、确定 | 事实问答、法律、技术文档 |
| **0.5~0.7** | 平衡、自然 | 日常对话、客服 |
| **0.8~1.0** | 标准随机 | 创意写作 |
| **1.5+** | 高度发散 | 头脑风暴 |

→ **Agent 用 0.0~0.3** 最稳。`temperature=0` 让工具调用永远确定。

### 旋钮 2:Top-k

**思路**:只在概率前 k 个 token 里采样,其他全归零。

- `top_k=1` → 等价于贪心(完全确定)
- `top_k=50` → 防止采到极低概率的"垃圾"词

### 旋钮 3:Top-p(Nucleus Sampling)

**思路**:按概率从高到低累加,**直到累积 ≥ p**,只在这个集合里采样。

- `top_p=0.9` 常用
- **比 Top-k 更智能**:动态适应分布

### ⭐ 三者优先级

```
温度 → Top-k → Top-p (按顺序应用)
```

**实战建议**:
- Agent 工具调用:`temperature=0.0`,Top-k/Top-p 无所谓
- 对话生成:`temperature=0.7, top_p=0.9`
- 创意写作:`temperature=1.0, top_p=0.95`

→ 详细对比:[Happy-LLM 5.3.6 采样策略](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/11-%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC.md#-%E9%87%87%E6%A0%B7%E7%AD%96%E7%95%A5%E8%AF%A6%E8%A7%A3)

---

## 📝 3.2.2 提示工程(Agent 的"灵魂")

### Agent ≈ 80% 是 Prompt 工程

LLM Agent 的核心代码其实就是 **System Prompt**(见 [第 1 章实战](../01-%E7%AC%AC1%E7%AB%A0-%E5%88%9D%E8%AF%86%E6%99%BA%E8%83%BD%E4%BD%93/03-%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0%E6%97%85%E8%A1%8C%E5%8A%A9%E6%89%8B.md))。
**Prompt 写得好坏决定 Agent 成败**。

### 4 种基础提示模式

#### Zero-shot(零样本)
直接下指令,不给示例:
```
判断情感:Datawhale 的 AI Agent 课程非常棒!
```

#### One-shot(单样本)
给 1 个示例:
```
文本: 这家餐厅服务太慢了。
情感: 负面

文本: Datawhale 的 AI Agent 课程非常棒!
情感:
```

#### Few-shot(少样本)
给多个示例(尤其有正负样本)→ **效果通常最好**。

#### CoT(Chain-of-Thought,思维链)
**让模型"一步步想"**,推理能力大幅提升:
```
问题: 一个篮球队 80 场比赛胜率 60%,接下来 15 场赢 12 场,
      两个赛季总胜率多少?
请一步一步地思考并解答。
```

LLM 会输出:
```
第一步: 第一赛季赢 = 80 × 60% = 48 场
第二步: 总场数 = 80 + 15 = 95;总胜数 = 48 + 12 = 60
第三步: 总胜率 = 60/95 ≈ 63.16%
```

→ **CoT = LLM 涌现能力的核心**,做 Agent 必懂。
→ 详见 [Happy-LLM CoT](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02-LLM%E5%9B%9B%E5%A4%A7%E8%83%BD%E5%8A%9B.md#-%E8%83%BD%E5%8A%9B-4%E9%80%90%E6%AD%A5%E6%8E%A8%E7%90%86step-by-step-reasoning-cot)

### 高级技巧

| 技巧 | 用法 |
|---|---|
| **角色扮演** | "你是资深 Python 专家" → 调整语气和深度 |
| **上下文示例** | = Few-shot 的具体应用 |
| **结构化输出** | 让 LLM 输出 JSON / XML(Agent 必需) |
| **指令调优模型 vs 基础模型** | 现代 Agent 全用**指令调优**版(ChatGPT/Qwen-Chat 等) |

### ⭐ 指令调优(Instruction Tuning)的革命

- **早期 GPT-3**(基础模型):需要 Few-shot 才能用
- **现代 ChatGPT/Qwen/Claude**:**指令调优后**,**直接下指令就行**

→ 做 Agent **永远用指令调优版**,不要用基础模型。
→ 详见 [Happy-LLM SFT 详解](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-SFT%E6%9C%89%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md)

---

## 🔤 3.2.3 文本分词(影响 Agent 成本的关键)

### 为啥 Agent 开发者要懂分词?

| 影响 | 解释 |
|---|---|
| **上下文窗口** | LLM 上下文是按 **Token** 数算的(不是字数) |
| **API 成本** | API 按 Token 计费,**算错预算就破产** |
| **模型表现** | `2+2`(无空格)可能被切成奇怪 Token,**导致计算错误** |

### 三种分词方式

| 方式 | 例子 | 痛点 |
|---|---|---|
| **按词** | "Datawhale_Agent" | 词表爆炸 + OOV |
| **按字符** | "D,a,t,a,w,h,a,l,e" | 序列太长 |
| **子词** ⭐ | "Data + whale + _Agent" | **平衡** |

→ **现代 LLM 全部用子词**。

### 主流子词算法:BPE

> BPE = Byte-Pair Encoding,GPT 系列用的算法。

**核心思想**:贪心合并最高频的相邻字符对。

```
初始: ['h', 'u', 'g', 'p', 'u', 'g', ...]
   ↓ 合并 ('u', 'g') 最高频
新增: 'ug'
   ↓ 合并 ('ug', '</w>')
新增: 'ug</w>'
   ↓ 继续...
```

→ 详细 BPE 算法 + 代码实现:[Happy-LLM 5.2 训练 Tokenizer](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/07-%E8%AE%AD%E7%BB%83Tokenizer.md)

### 其他算法

| 算法 | 用户 |
|---|---|
| **WordPiece** | BERT |
| **SentencePiece** | LLaMA |

### ⚠️ Token 的"坑"

#### 坑 1:中文 Token 通常比英文贵
- 英文 1 单词 ≈ 1 Token
- 中文 1 字 ≈ 1-2 Token
- 同样意思,**中文费 2-3 倍 Token**!

#### 坑 2:Agent 多轮对话 Token 累积
- 每轮把历史都塞进去 → **Token 飞涨**
- 解法:**用 Memory + 摘要压缩**(第 8 章会讲)

#### 坑 3:格式细节影响 Token
- `2+2` vs `2 + 2` → **切成不同 Token**
- 数字、特殊符号要小心

---

## 🐍 3.2.4 调用开源 LLM(本地 / Hugging Face)

### 为啥要本地跑?

| 场景 | 用 API | 用本地 |
|---|---|---|
| 快速原型 | ✅ | – |
| 敏感数据 | – | ✅ |
| 离线运行 | – | ✅ |
| 成本极致控制 | – | ✅ |
| 低延迟 | – | ✅ |

### 用 Hugging Face Transformers(标准做法)

```python
# 安装
# pip install transformers torch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载小模型(0.5B,大多数电脑能跑)
model_id = "Qwen/Qwen1.5-0.5B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# 2. 准备对话(用 chat_template 标准化)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好,请介绍你自己。"}
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 3. 编码 → 生成 → 解码
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

→ **3 步走**:加载 → 编码 → 生成。

### 模型选择建议

| 资源               | 推荐                               |
| ---------------- | -------------------------------- |
| 笔记本(无 GPU)       | **Qwen-1.5B / 0.5B**(本节示例)       |
| 单卡 24G(RTX 4090) | **Qwen2.5-7B / Llama-3-8B**      |
| 多卡集群             | **Qwen2.5-72B / DeepSeek-V3**    |
| 纯调 API           | **OpenAI GPT-4 / Claude / 通义千问** |

→ 模型选型详细对比:[Happy-LLM 3.x 选型决策树](../../Happy-LLM/03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)

---

## 🎯 做 Agent 时模型选型的 3 大权衡

```
能力 ←─→ 成本 ←─→ 速度

最强(GPT-4) ─ 最贵 ─ 慢
中等(GPT-4-mini) ─ 平衡 ─ 中
便宜(开源 7B) ─ 几乎免费 ─ 快(本地)
```

**实战建议**:
- **MVP 阶段**:用 GPT-4 / Claude(质量优先)
- **规模化**:换便宜 API(GPT-4-mini / DeepSeek)
- **极致控成本**:开源模型 + 本地部署

---

## ⚠️ 小白避坑

1. **`temperature=0` 不代表完全可复现**
   - GPU 浮点运算有微小随机性
   - 想完全可复现需设 `seed`
2. **`max_new_tokens` 别设太大**
   - 越大越贵越慢
   - 一般 200~500 够
3. **`stop` 参数很重要**
   - Agent 输出格式化文本时,设 `stop=["\nObservation:"]` 等
   - 避免 LLM 自己脑补 Observation
4. **不要在 prompt 里塞太多无关上下文**
   - 浪费 Token + 干扰理解

---

## 📌 3.2 节要点(必背)

| 知识点 | 一句话 |
|---|---|
| **Temperature** | 控制随机性,Agent 用 0~0.3 |
| **Top-k / Top-p** | 候选集筛选,Top-p 更智能 |
| **Zero/One/Few-shot** | 给几个示例的差别 |
| **CoT** | 一步步思考,推理能力暴涨 |
| **指令调优** | 现代 LLM 标配,直接下指令 |
| **BPE 分词** | 影响 Token 数和成本 |
| **中文比英文贵** | 同样意思 Token 多 2~3 倍 |

---

## 🔗 延伸阅读

- 上一节:[01-LLM架构速览](01-LLM%E6%9E%B6%E6%9E%84%E9%80%9F%E8%A7%88.md)
- 下一节:[03-缩放法则与局限](03-%E7%BC%A9%E6%94%BE%E6%B3%95%E5%88%99%E4%B8%8E%E5%B1%80%E9%99%90.md)
- 完整 Tokenizer:[Happy-LLM 5.2](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/07-%E8%AE%AD%E7%BB%83Tokenizer.md)
- 完整生成与采样:[Happy-LLM 5.3.6](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/11-%E7%94%9F%E6%88%90%E6%96%87%E6%9C%AC.md)

---

⬅ [01-LLM架构速览](01-LLM%E6%9E%B6%E6%9E%84%E9%80%9F%E8%A7%88.md)	|	➡ [03-缩放法则与局限](03-%E7%BC%A9%E6%94%BE%E6%B3%95%E5%88%99%E4%B8%8E%E5%B1%80%E9%99%90.md)
