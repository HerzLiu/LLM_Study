---
tags: [Hello-Agents, 第11章, 理论, PBRFT, Agentic-RL, MDP]
chapter: 11
section: 11.1
---

# 11.1 从 LLM 训练到 Agentic-RL

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-数据集与奖励函数](02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)

---

## 🎬 故事比喻:从"鹦鹉学舌"到"自主进化"

| 阶段 | 比喻 |
|---|---|
| **预训练** | 鹦鹉听了海量人类对话(学会"说话") |
| **SFT** | 鹦鹉跟训练师学固定回答(学会"按指令回话") |
| **RLHF** | 鹦鹉根据观众反馈调整(学会"说大家爱听的") |
| **Agentic-RL** ⭐ | 鹦鹉**自己探索新表达方式**(**超越训练数据**) |

→ Agentic-RL 让 Agent 从"**模仿者**"变成"**自主探索者**"。

---

## 🏗 11.1.1 LLM 训练全景图(2 阶段)

```
┌─────────────────────────────────────┐
│  阶段 1: 预训练 (Pretraining)         │
│  ──────────────────────────         │
│  数据: 数 TB 互联网文本               │
│  任务: 预测下一个词 (CLM)             │
│  产出: 通用语言能力                   │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│  阶段 2: 后训练 (Post-training)        │
│  ──────────────────────────         │
│  Step 1: SFT (监督微调)               │
│     - 学指令遵循 + 对话格式            │
│  Step 2: RM (奖励建模)                │
│     - 学人类偏好                       │
│  Step 3: RL 微调 (PPO/DPO/GRPO)       │
│     - 优化生成质量                     │
└─────────────────────────────────────┘
```

→ **完整的"语言模型 → 对话助手"演化路径**。

> 📚 完整原理详见:[Happy-LLM 三阶段训练](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04-%E8%AE%AD%E7%BB%83%E4%B8%89%E9%98%B6%E6%AE%B5%E5%85%A8%E6%99%AF.md)

---

## 📚 11.1.2 预训练 vs 后训练

### 预训练:**学语言 + 学世界**

**核心任务**:**预测下一个词**(Causal Language Modeling)

```
给定: "The cat sat on the"
预测: "mat" (最可能的下一个词)
```

**目标函数**(最大化对数似然):

$$\mathcal{L}_{\text{pretrain}} = \sum_{t} \log P_\theta(x_{t+1} | x_1, \dots, x_t)$$

**特点**:
- 数据量**巨大**(TB 级)
- 计算成本**极高**(千卡 × 月)
- **无监督学习**
- 产出**通用语言能力**

### 后训练:**学指令 + 对齐人类**

#### Step 1:**SFT**(监督微调)

```
训练数据: (prompt, completion) 对
目标: 学会"听指令 + 按格式回话"
```

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log P_\theta(y_i | x_i)$$

#### Step 2:**RM**(奖励建模)

```
训练数据: 偏好对 (chosen vs rejected)
目标: 学会"判断什么回答更好"
```

$$\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

#### Step 3:**RL 微调**(PPO/DPO/GRPO)

```
目标: 用 RM 评分作为奖励,优化策略
```

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}[r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})]$$

→ **关键技巧:KL 散度限制**,防止模型偏离原始太远。

---

## 🆕 11.1.3 RLAIF:用 AI 替代人类标注

传统 RLHF 需要**大量人工标注偏好对**,**成本高**。

**RLAIF** 思路:**用强大 AI(如 GPT-4)替代人类标注**。

```
1. SFT 模型生成多个候选回答
   ↓
2. 强大 AI 评分 + 排序
   ↓
3. 用 AI 评分训练 RM
   ↓
4. 用 RM 进行 RL
```

**实验结果**:**接近甚至超过 RLHF**,**成本大幅降低**。
→ 这是 Anthropic Constitutional AI 等的核心思想。

---

## 🤖 11.1.4 PBRFT vs Agentic-RL(本章核心)

### 两种范式

| | **PBRFT**(单轮) | **Agentic-RL**(多步) |
|---|---|---|
| 全称 | Preference-Based RFT | Agentic Reinforcement Learning |
| 关注 | **单轮回答质量** | **多步任务完成** |
| 适合 | 对话助手 | **自主智能体** |

### 例子对比

#### PBRFT 场景
```
用户: "请解释什么是强化学习"
   ↓
模型生成完整回答
   ↓
RM 给分 → 单步奖励
```

#### Agentic-RL 场景
```
用户: "帮我分析这个 GitHub 仓库代码质量"
   ↓
Step 1: 调 GitHub API → +0.1 奖励
Step 2: 读主要代码文件 → +0.1
Step 3: 分析代码质量 → +0.2
Step 4: 生成报告 → +0.6
─────────────────────────
总奖励: 1.0 (累积)
```

---

## 🧮 11.1.5 MDP 框架对比(必背)

> **MDP** = (S, A, P, R, γ) —— 强化学习的标准框架

### PBRFT 形式化

| 维度 | PBRFT |
|---|---|
| **状态 S** | 仅用户提示 |
| **行动 A** | 仅文本生成 |
| **转移 P** | 无(单步) |
| **奖励 R** | 单步,仅任务结束时 |
| **时间** | T=1 |
| **目标** | $\max \mathbb{E}[R(s, a)]$ |

### Agentic-RL 形式化

| 维度 | Agentic-RL |
|---|---|
| **状态 S** | 历史观察 + 上下文 |
| **行动 A** | 文本生成 + 工具调用 + 环境操作 |
| **转移 P** | 状态根据行动动态变化 |
| **奖励 R** | **多步**,可中间步骤给奖励 |
| **时间** | T > 1(多步) |
| **目标** | $\max \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$ |

### 关键差异:**轨迹(Trajectory)**

```
PBRFT:  (s, a, r)               单一三元组

Agentic-RL: τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)
                完整轨迹
```

→ Agentic-RL 优化**整条轨迹**,不是单步。

---

## 🌟 11.1.6 Agentic-RL 的 6 大核心能力

```
                Agentic-RL 赋予 LLM 的 6 大能力
                          │
        ┌───────┬─────────┼─────────┬───────┐
        │       │         │         │       │
     推理      工具       记忆      规划   自我改进  感知
     │       │         │         │       │       │
  超越 CoT  学会用    管理       动态     反思    多模态
  自主探索  哪些工具  上下文     规划     纠错    理解
            何时用    什么时候   权衡
                      记 / 忘    短长期
```

### 能力 1:**推理(Reasoning)** ⭐ 最重要

**传统方法的局限**:
- CoT 提示:依赖少样本,**泛化差**
- SFT:**只能模仿**训练数据的推理模式

**RL 的优势**:
- 通过**试错**学有效推理策略
- **发现训练数据中没有的推理路径**(如 DeepSeek-R1 的"啊哈时刻")
- 学会**何时深度思考 / 何时快速回答**

**形式化**:
$$\mathcal{R}(q, c, a) = 1 \text{ if } a = a^* \text{ else } 0$$
$$\max_\theta \mathbb{E}_{(c, a) \sim \pi_\theta(q)}[R]$$

### 能力 2:**工具使用(Tool Use)**

```
行动空间扩展:
   A = {a_text, a_tool}
       ↑           ↑
   文本生成   工具调用
```

RL 让 Agent 学会:
- 何时**需要**工具
- 选**哪个**工具
- 如何**组合**多个工具

### 能力 3:**记忆(Memory)**

```
RL 让 Agent 学:
- 哪些信息值得记
- 何时更新记忆
- 何时遗忘
→ 类似人类工作记忆
```

### 能力 4:**规划(Planning)**

```
传统 CoT: 线性思考,无法回溯
RL 后:    动态规划,试错发现有效序列
         + 权衡短期 vs 长期收益
```

### 能力 5:**自我改进(Self-Improvement)**

```
- 反思自己输出
- 识别错误
- 调整策略
→ 类似"从错误学习"
```

### 能力 6:**感知(Perception)**

```
- 多模态理解
- 学会使用视觉工具
- 视觉规划
```

---

## 🏗 11.1.7 HelloAgents Agentic-RL 设计

### 技术选型

| 组件 | 选择 | 原因 |
|---|---|---|
| **RL 框架** | **TRL**(Hugging Face) | 成熟稳定,功能完整 |
| **基础模型** | **Qwen3-0.6B** | 小模型适合普通 GPU |
| **数据集** | **GSM8K** | 数学推理,客观可评估 |
| **算法** | SFT + **GRPO** | GRPO 比 PPO 简化 |

### 4 层架构

```
┌─────────────────────────────────────┐
│  4. 统一接口层                       │
│     RLTrainingTool                   │
│     action: train/load_dataset/...   │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  3. 训练器层                         │
│     SFTTrainerWrapper                │
│     GRPOTrainerWrapper               │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  2. 奖励函数层                       │
│     AccuracyReward (基础)             │
│     LengthPenaltyReward (惩罚)        │
│     StepReward (步骤奖励)             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  1. 数据集层                         │
│     GSM8KDataset                     │
│     create_sft_dataset()             │
│     create_rl_dataset()              │
└─────────────────────────────────────┘
```

---

## 🚀 11.1.8 30 秒体验

```bash
# 安装(第 11 章版本)
pip install "hello-agents[rl]==0.2.5"
```

```python
from hello_agents.tools import RLTrainingTool
import json

rl_tool = RLTrainingTool()

# 1. SFT 训练(10 样本,快速测试)
sft_result = rl_tool.run({
    "action": "train",
    "algorithm": "sft",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_sft",
    "max_samples": 10,
    "num_epochs": 1,
    "use_lora": True,
})

# 2. GRPO 训练
grpo_result = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./models/quick_grpo",
    "max_samples": 5,
    "num_epochs": 1,
    "use_lora": True,
})

# 3. 评估
eval_result = rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/quick_grpo",
    "max_samples": 10,
})
print(f"准确率: {json.loads(eval_result)['accuracy']}")
```

→ **3 步跑通**完整训练流程。
> ⚠️ **快速测试准确率会很低**(模型只见过 0.7% 数据)。

---

## ⚠️ 小白避坑

1. **不要跳过 SFT 直接 GRPO**
   - 没 SFT 基础,**RL 经常崩**
2. **小模型 + LoRA = 入门标配**
   - 8GB GPU 也能跑
3. **GSM8K 之外的任务?**
   - 自定义数据集 + 奖励函数(11.2 节)
4. **训练慢 + 显存爆?**
   - 减小 `batch_size` / `max_samples`
   - 用更小的 `lora_rank`

---

## 📌 11.1 节要点

| 知识点 | 一句话 |
|---|---|
| **LLM 训练 2 阶段** | 预训练(学语言)+ 后训练(对齐人类) |
| **后训练 3 步** | SFT → RM → RL |
| **RLAIF** | 用 AI 替代人工标注,**降本** |
| **PBRFT vs Agentic-RL** | 单步质量 vs **多步任务完成** |
| **MDP 差异** | 状态/行动/奖励 都变成**多步** |
| **6 大核心能力** | **推理/工具/记忆/规划/自我改进/感知** |
| **技术栈** | TRL + Qwen3-0.6B + GSM8K |

---

## 🔗 延伸阅读

- 上一节:[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节:[02-数据集与奖励函数](02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)
- **必读前置**:[Happy-LLM RLHF 详解](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)
- 跨书:[Happy-LLM 三阶段](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/04-%E8%AE%AD%E7%BB%83%E4%B8%89%E9%98%B6%E6%AE%B5%E5%85%A8%E6%99%AF.md)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-数据集与奖励函数](02-%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8E%E5%A5%96%E5%8A%B1%E5%87%BD%E6%95%B0.md)
