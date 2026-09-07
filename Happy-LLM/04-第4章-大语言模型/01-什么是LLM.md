---
tags: [Happy-LLM, 第4章, LLM, 定义]
chapter: 4
section: 4.1.1
---

# 4.1.1 什么是 LLM？—— 和 PLM 的本质区别

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-LLM四大能力](02-LLM%E5%9B%9B%E5%A4%A7%E8%83%BD%E5%8A%9B.md)

---

## 🎬 故事比喻：村里的天才 vs 全国冠军

[第 3 章](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) 讲的 **PLM**（[BERT](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/02-BERT.md)、[GPT-1](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-GPT.md)）就像**村里的天才**：
- 能在某些题目上拿高分
- 但每换一种题型都要单独教（**微调**）
- 总有解不了的复杂题

**LLM**（GPT-4、LLaMA、Claude）则是**全国冠军**：
- 见过所有题型
- 你描述一下任务它就会做（**ICL**）
- 复杂的数学题、写代码、写论文都能做
- 还会"举一反三"（**涌现能力**）

→ 区别不是"大一点的 BERT"，而是**质的飞跃**。

---

## 🔧 LLM 的定义

> **LLM = Large Language Model（大语言模型）**
> 一种**参数量更大、训练语料更海量**的语言模型，展现出**与传统 PLM 截然不同的能力**。

### 三个关键修饰词

| 词 | 解释 |
|---|---|
| **Large**（大） | 参数量 ≥ 数百亿（广义可下探到 10 亿） |
| **Language**（语言） | 处理自然语言文本 |
| **Model**（模型） | 通常是 [Decoder-only](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-GPT.md) 架构 + [CLM 预训练任务](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-GPT.md#-%E9%A2%84%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1clm%E5%9B%A0%E6%9E%9C%E8%AF%AD%E8%A8%80%E5%BB%BA%E6%A8%A1) |

### 严格 vs 宽松定义

| 标准 | 参数下限 | 例子 |
|---|---|---|
| **严格** | ≥ 100B | GPT-3 (175B)、GPT-4、Claude |
| **宽松** | ≥ 1B | Qwen-1.8B、Phi-3、LLaMA-7B |
| **本质** | 展现**涌现能力**就算 | – |

→ 真正的判定标准不是参数量，而是**涌现能力**：
> 在一系列复杂任务上**远超传统 PLM** 的能力 → 就是 LLM。

---

## 🆚 LLM vs PLM 对比表

| 维度 | 传统 PLM（如 BERT） | **LLM**（如 GPT-4） |
|---|---|---|
| 参数量 | 0.1B – 0.3B | **10B – 千亿+** |
| 训练 token | 3B – 30B | **数 T 甚至数十 T** |
| 架构 | Encoder-only / Encoder-Decoder | **Decoder-only**（主流） |
| 适配下游任务 | 必须微调 | **直接用 ICL** |
| 训练成本 | 几十 GPU × 几天 | **千卡级别 × 数月** |
| 数据成本 | 高（每任务标几千条） | 极低（写几个 prompt） |
| 通用性 | 一个任务一个模型 | **一个模型搞定一切** |
| 涌现能力 | ❌ 无 | ✅ 有 |
| 价格 | 可本地跑 | 需云算力 |

---

## 📅 LLM 时代关键时间线

```
2020.05 ─── GPT-3（175B，LLM 时代开端）
2022.11 ─── ChatGPT 横空出世
2023.02 ─── LLaMA-1 开源（开源 LLM 元年）
2023.03 ─── GPT-4 / Claude / Bard / 文心一言 / ChatGLM
2023.04~05 ─ 通义千问 / Falcon / PaLM-2 / 星火 / Pi
2023.06~07 ─ ChatGLM2 / Baichuan / LLaMA-2 / Claude-2 / 盘古-3
2023.08~11 ─ 豆包 / Gemini / 混元 / Yi / DeepSeek / Grok
2024.04 ─── LLaMA-3、Qwen-2、Phi-3 等持续迭代
...
```

> 一句话：**2 年时间冒出上百个 LLM**，国内国外、开源闭源、专业通用、各种规模一应俱全。

---

## ⭐ 加深理解：什么是「涌现」？

「涌现」是 LLM 区别于 PLM 的**决定性特征**，单独一节详细讲：

→ [涌现能力详解](02-LLM%E5%9B%9B%E5%A4%A7%E8%83%BD%E5%8A%9B.md#-%E8%83%BD%E5%8A%9B-1%E6%B6%8C%E7%8E%B0%E8%83%BD%E5%8A%9Bemergent-abilities)

简单说就是：
- 小模型完全不会的任务
- 模型 ↑ 到某个阈值后**突然就会了**
- 类似**水沸腾的相变**

---

## ⚠️ 小白避坑

1. **LLM 不等于 ChatGPT**
   - ChatGPT 是 LLM 的一种 + 对话产品包装
   - 实际有很多 LLM（GPT-4、Claude、LLaMA、Qwen…）
2. **不是所有"AI 产品"都是 LLM**
   - 智能音箱、人脸识别等不是
   - LLM 特指**语言模型**
3. **小模型也叫 LLM 没问题**
   - Qwen-1.8B、Phi-3-mini 等"小 LLM"也很受欢迎
   - 在终端部署、低成本场景有优势
4. **LLM 不等于 AGI**
   - LLM 是通往 AGI 的**一条路**，但还没到
   - 推理、可控性、幻觉等问题待解决

---

## 🌍 LLM 大家族（按 2024 年视角）

### 闭源系（API 商用）
- **OpenAI**：GPT-3.5 / GPT-4 / GPT-4o
- **Anthropic**：Claude 3 系列
- **Google**：Gemini
- **百度**：文心一言
- **科大讯飞**：星火
- **腾讯**：混元
- **字节**：豆包
- **商汤**：日日新

### 开源系（可本地部署）
- **Meta**：LLaMA-1/2/3
- **阿里**：Qwen / Qwen2 / Qwen2.5
- **DeepSeek**：DeepSeek-V2 / V3 / R1
- **智谱**：ChatGLM / GLM-4-9B
- **百川**：Baichuan
- **零一万物**：Yi
- **微软**：Phi
- **Mistral AI**：Mistral / Mixtral

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-LLM四大能力](02-LLM%E5%9B%9B%E5%A4%A7%E8%83%BD%E5%8A%9B.md) —— LLM 的"魔法"
- 架构基础：[06-GPT](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/06-GPT.md)、[07-LLaMA](../03-%E7%AC%AC3%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-LLaMA.md)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-LLM四大能力](02-LLM%E5%9B%9B%E5%A4%A7%E8%83%BD%E5%8A%9B.md)
