---
tags: [Happy-LLM, 第3章, PLM, GLM, ChatGLM, 中文LLM]
chapter: 3
section: 3.3.3
---

# 3.3.3 GLM —— 中文开源先锋

⬅ [07-LLaMA](07-LLaMA.md)	|	➡ [09-横向对比与选型](09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)

> 📅 清华智谱 2023.03 发布 ChatGLM-6B（**国内首个开源中文 LLM**）
> 🎯 核心信息：**用独特的"GLM 预训练任务"，试图统一 NLU 和 NLG**

---

## 🎬 故事比喻：左右逢源的尝试

[BERT](02-BERT.md) 是 NLU 派（完形填空，左右都看）。
[GPT](06-GPT.md) 是 NLG 派（写续作，只看左）。

**GLM 团队想**："能不能我两个都要？"
→ 设计了一种**结合自编码（MLM）和自回归（CLM）**的预训练任务，叫 **GLM 任务**。

虽然在 LLM 时代这条路最终被证明不如纯 CLM，但 GLM 系列仍然是**中文 LLM 的奠基者**之一。

---

## 🏗 模型架构：基本是 GPT，三点小改动

GLM 和 [GPT](06-GPT.md) 几乎一样（都是 Decoder-only），但有三个细微差异：

### 差异 1：Post-Norm 而非 Pre-Norm

| 类型 | 顺序 |
|---|---|
| **Pre-Norm**（主流） | LayerNorm → Sublayer → 残差 |
| **Post-Norm**（GLM 选择） | Sublayer → 残差 → LayerNorm |

> 详见 [Pre-Norm vs Post-Norm 对比](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/08-%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5.md#-pre-norm-vs-post-norm)

| 优势 | 谁占优 |
|---|---|
| 训练稳定性 | Pre-Norm |
| 参数正则化效果 | Post-Norm |
| 防梯度爆炸/消失 | Pre-Norm |

GLM 论文认为 **Post-Norm 能避免 LLM 的"数值错误"**。但**主流 LLM 仍然选 Pre-Norm**，包括 ChatGLM2 起也回归了 Pre-Norm。

### 差异 2：用单线性层替代 MLP 输出

- GPT 最后输出用 MLP（两层线性）
- GLM 用**单个线性层**直接预测 token
- → 减少最终输出参数，**把参数预算放到模型主体**

### 差异 3：激活函数 GeLU 替代 ReLU

> ReLU 在 0 处不连续；GeLU 平滑且更适合深层网络

---

## 🎯 预训练任务：GLM 任务（核心创新）

### 思想：自编码（MLM）+ 自回归（CLM）融合

#### 怎么融合？

1. **像 MLM 一样**随机遮蔽 token
2. 但**不是遮蔽单个 token，而是连续一段（span）**
3. **遮蔽部分内部用 CLM 顺序生成**

#### 例子

```
原文：I love you because you are a wonderful person

随机遮蔽连续两段：
  I <MASK> because you <MASK>

GLM 任务要求模型：
  <MASK_1> → love you
  <MASK_2> → are a wonderful person

具体生成时是 token by token（CLM 风格）：
  <MASK_1>: l → o → v → e → ...
```

#### 效果

| 任务类型       | GLM 表现           |
| ---------- | ---------------- |
| **NLG 任务** | ✅ 适合（内部用 CLM 生成） |
| **NLU 任务** | ✅ 适合（看到了双向上下文）   |

→ GLM 模型在**同体量下确实超过 BERT 系**，证明思路可行。

### ⭐ GLM 任务 vs T5 的 Span Corruption

| | **GLM** | [T5 Span Corruption](05-T5.md#-%E9%A2%84%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1) |
|---|---|---|
| 架构 | Decoder-only | Encoder-Decoder |
| 遮蔽对象 | 连续 span | 连续 span |
| 生成方式 | 在统一序列中 CLM 生成 | Decoder 输出生成 |

两者**思想接近**，只是架构不同。

---

## 📈 ChatGLM 家族演进

### ChatGLM-6B（2023.03）

| 项 | 数值 |
|---|---|
| 参数 | 6B |
| 预训练 token | 1T |
| 上下文 | 2K |
| **重要意义** | **国内首个开源中文 LLM**，影响巨大 |

参考 ChatGPT 加入 **SFT + RLHF**，成为众多中文 LLM 研究者的起点。

### ChatGLM2-6B（2023.06）

| 项 | 改进 |
|---|---|
| 上下文 | **32K**（大幅扩展） |
| 架构 | **回归 LLaMA 风格**（Pre-Norm 等） |
| 注意力 | 引入 **MQA**（Multi-Query Attention） |
| 预训练任务 | **回归经典 CLM**（放弃 GLM 任务） |

→ **ChatGLM2 标志着 GLM 任务"试验失败"**，承认 CLM 在大规模下更优。

### ChatGLM3-6B（2023.10）

| 项 | 改进 |
|---|---|
| 架构 | 与 ChatGLM2 相同 |
| 性能 | 中文场景 SOTA |
| 新增功能 | **支持 function calling + 代码解释器** |

→ 直接支持开发 Agent 应用。

### GLM-4（2024.01）

| 项 | 改进 |
|---|---|
| 上下文 | **128K** |
| 性能 | 英文基准达到 **GPT-4 水平** |
| 开源情况 | GLM-4 闭源，但**开源了 GLM-4-9B 轻量版** |

GLM-4-9B 在 1T 多语言语料预训练，超越 LLaMA-3-8B，**支持所有 GLM-4 工具功能**。

---

## ⭐ 加深理解：GLM 任务为什么最终被放弃？

虽然 GLM 任务设计巧妙，但在 LLM 时代输给了 CLM：

| 原因 | 解释 |
|---|---|
| **CLM 规模化更好** | 大模型下 CLM 的生成能力反而能"带飞"理解能力 |
| **GLM 实现复杂** | 训练时要做 span 遮蔽和重排，工程麻烦 |
| **预训练数据利用率** | CLM 每个 token 都是训练信号；GLM 只有被遮蔽的部分 |
| **下游适配性** | 现代 LLM 都做 instruction tuning，CLM 更直接 |

→ 这印证了 [GPT 章节](06-GPT.md)里的观点：**架构越简单，规模化越容易，最终越赢**。

不过 GLM 思想仍有借鉴意义：
- T5 的 Span Corruption 是类似思路
- 某些**多任务训练**场景下 GLM 任务仍然有效

---

## ⚠️ 小白避坑

1. **ChatGLM ≠ GLM 任务**：
   - 只有 **ChatGLM-1** 用 GLM 任务
   - **ChatGLM-2 起回归 CLM**
2. **GLM-4 不开源**，但 GLM-4-9B 开源
3. **国内 LLM 排行**：
   - Qwen（阿里）、ChatGLM（智谱）、DeepSeek、Baichuan、Yi 各有所长
   - 都基本采用 LLaMA 风格架构
4. **学习 GLM 思想 vs 实际用 GLM**：
   - 思想（自编码 + 自回归融合）有学习价值
   - 实际用 LLM 用 LLaMA 风格架构即可

---

## 🌏 中文 LLM 生态速览

| 模型           | 开发方       | 特点                 |
| ------------ | --------- | ------------------ |
| **ChatGLM**  | 智谱        | 中文 LLM 先驱、Agent 友好 |
| **Qwen**     | 阿里        | 多尺寸全面，技术报告详尽       |
| **Baichuan** | 百川智能      | 早期开源活跃             |
| **Yi**       | 零一万物      | 李开复团队              |
| **Skywork**  | 昆仑万维      | 多模态                |
| **DeepSeek** | 深度求索      | 高性价比、MoE 架构        |
| **InternLM** | 上海 AI Lab | 全面开源               |

---

## 🔗 延伸阅读

- 上一节：[07-LLaMA](07-LLaMA.md) —— GLM 借鉴的对象
- 下一节：[09-横向对比与选型](09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md) —— 全章总结
- 第 4 章：[LLM 概念](../04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

⬅ [07-LLaMA](07-LLaMA.md)	|	➡ [09-横向对比与选型](09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)
