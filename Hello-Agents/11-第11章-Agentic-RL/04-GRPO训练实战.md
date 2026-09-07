---
tags: [Hello-Agents, 第11章, GRPO, PPO, RL训练]
chapter: 11
section: 11.4
---

# 11.4 GRPO 训练实战

⬅ [03-SFT训练实战](03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)	|	➡ [05-端到端流程与小结](05-%E7%AB%AF%E5%88%B0%E7%AB%AF%E6%B5%81%E7%A8%8B%E4%B8%8E%E5%B0%8F%E7%BB%93.md)

> **DeepSeek-R1** 训练用的算法,**RL 训练 LLM 的当下最佳实践**。

---

## 🎬 故事比喻:从"PPO 全家桶"到"GRPO 极简"

| | **PPO**(传统) | **GRPO**(简化) |
|---|---|---|
| 模型数 | **4 个**(Policy + Reference + Value + Reward) | **2 个**(Policy + Reference) |
| 显存 | 巨大 | **大幅降低** |
| 复杂度 | 高 | **低** |
| 稳定性 | 易崩 | **更稳** |

→ **GRPO = PPO 的"瘦身版"**,专为 LLM 设计。

---

## ❌ 11.4.1 PPO 的痛点

### 痛点 1:**需要 Value Model**

```
PPO 训练要 4 个模型同时在显存:
1. Policy Model        (要训的)
2. Reference Model     (KL 约束)
3. Value Model         (估算未来累积奖励)
4. Reward Model        (评分)

→ 显存爆炸,工程复杂
```

### 痛点 2:**训练不稳定**

```
- 奖励崩塌
- 策略退化
- 各种调参噩梦
```

→ DeepSeek 团队**忍无可忍**,搞出了 GRPO。

> 📚 PPO 详解见 [Happy-LLM PPO](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md#-%E7%AC%AC%E2%91%A1%E6%AD%A5ppoproximal-policy-optimization%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96)

---

## ✨ 11.4.2 GRPO 的核心思想

### 核心创新:**用"组内相对奖励"代替"绝对奖励"**

```
PPO 思路:
   每个回答给个绝对奖励 R(s, a)
   再用 Value Model 估算优势函数 A(s, a)

GRPO 思路:
   对同一个问题生成 G 个回答 (a_1, ..., a_G)
   计算每个回答的奖励 r_i
   ⭐ 优势 = (r_i - mean(r)) / std(r)
   → 不需要 Value Model!
```

### 数学形式化

#### 第一步:**采样 G 个回答**

对每个问题 $q$,从当前策略采样 G 个回答:

$$\{o_1, o_2, \dots, o_G\} \sim \pi_\theta(\cdot | q)$$

#### 第二步:**计算每个的奖励**

$$r_i = R(q, o_i), \quad i = 1, \dots, G$$

#### 第三步:**计算优势**(组内归一化)

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$$

→ **优势 = "我比组内平均好多少"**(标准化分数)。

#### 第四步:**GRPO 目标函数**

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}_q \left[ \frac{1}{G} \sum_{i=1}^G \min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) \right] - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

其中:
- $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}$:重要性采样比
- `clip`:**PPO 的核心**,防止更新过大
- $\beta \cdot \text{KL}$:**防止偏离参考模型**

→ **PPO 的 clip 还在**,**只是去掉了 Value Model**。

---

## 🆚 11.4.3 GRPO vs PPO 完整对比

| 维度 | **PPO** | **GRPO** |
|---|---|---|
| **模型数** | 4 个 | **2 个** |
| **Value Model** | 需要 | **不需要** |
| **优势计算** | $A = Q - V$ | $A = \frac{r - \text{mean}}{\text{std}}$ |
| **采样方式** | 单次 | **组采样**(G 次) |
| **显存** | 大 | **小** |
| **稳定性** | 中 | **高** |
| **代表应用** | InstructGPT | **DeepSeek-R1** |

---

## 🚀 11.4.4 GRPO 实战:基础示例

```python
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

# GRPO 训练
result = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",

    # 模型 - 通常用 SFT 后的模型作为起点
    "model_name": "./models/sft_model",   # SFT 输出
    "output_dir": "./models/grpo_model",

    # 数据
    "max_samples": 1000,

    # GRPO 特定参数
    "num_generations": 8,         # ⭐ G = 8 (每问题采 8 个回答)
    "max_completion_length": 512, # 每回答最大长度
    "temperature": 0.7,           # 采样温度

    # 训练
    "num_epochs": 2,
    "batch_size": 2,              # ⭐ 必须能被 G 整除
    "learning_rate": 5e-6,        # ⭐ GRPO 用更小学习率

    # LoRA
    "use_lora": True,
    "lora_rank": 16,
})

print(f"GRPO 模型: {result['model_path']}")
```

---

## 🔧 11.4.5 GRPO 关键参数详解

### GRPO 特有参数

| 参数 | 含义 | 建议 |
|---|---|---|
| **`num_generations`** | 每问题采样数 G | **8** 是经典值,4-16 |
| `max_completion_length` | 每回答最大长度 | 256-1024 |
| `temperature` | 采样温度 | 0.7-1.0(要多样性) |
| `beta` | KL 散度系数 | 0.04-0.1(默认 0.04) |

### 关键约束:**`batch_size % G == 0`**

```python
# ✅ 合法
batch_size = 8,  G = 8   # 8 / 8 = 1
batch_size = 16, G = 8   # 16 / 8 = 2
batch_size = 4,  G = 4   # 4 / 4 = 1

# ❌ 错误
batch_size = 5,  G = 8   # 5 / 8 = 0.625
batch_size = 16, G = 5   # 16 / 5 = 3.2
```

→ 否则**报错!**

### 学习率:**比 SFT 小**

```
SFT:   5e-5  (大步前进)
GRPO:  5e-6  (小步精调)
```

**原因**:RL 容易把 SFT 的成果毁掉,**必须小心**。

---

## 🎯 11.4.6 完整 GRPO 训练示例

```python
# 完整 GRPO 训练
result = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",

    # 从 SFT 模型开始(必须!)
    "model_name": "./models/sft_full",
    "output_dir": "./models/grpo_full",

    # 数据
    "max_samples": None,           # 全部数据

    # GRPO 核心参数
    "num_generations": 8,
    "max_completion_length": 768,
    "temperature": 0.9,
    "beta": 0.04,                  # KL 系数

    # 训练
    "num_epochs": 2,
    "batch_size": 16,              # 必须能被 G=8 整除
    "learning_rate": 5e-6,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    # LoRA
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,

    # 奖励函数(默认用 AccuracyReward)
    # 或自定义:custom_reward=my_reward_fn

    # 监控
    "save_steps": 200,
    "logging_steps": 50,
})
```

→ **预计耗时**:**几小时到一天**(取决于硬件)。

---

## 📊 11.4.7 GRPO 训练监控

### 关键指标

| 指标 | 解读 |
|---|---|
| **平均奖励** | ⬆ = 模型在变好 |
| **奖励标准差** | 中等(0.1-0.5)= 探索多样性好 |
| **KL 散度** | < 0.1 = 没偏离原模型太远 |
| **学习率** | 按 warmup 策略变化 |

### 训练中的"啊哈时刻"(Aha Moment)

```
DeepSeek-R1 论文记录:
   GRPO 训练到某一时刻
   ↓
   模型自发学会:
   - 自我反思 ("Wait, let me reconsider...")
   - 错误纠正 ("Actually, I think...")
   - 多策略尝试
   ↓
   这是 RL 真正 超越训练数据 的标志!
```

→ **小模型很难出现 aha moment**,但**大模型(7B+)训得久会出现**。

---

## 🚨 11.4.8 常见问题与解决

| 问题 | 原因 | 解决 |
|---|---|---|
| **奖励崩塌** | 学习率太大 | 降到 1e-6 |
| **KL 散度爆炸** | β 太小 | 加大到 0.1 |
| **`batch_size` 不能被 G 整除** | 配置错 | 调整二者 |
| **OOM** | 模型 + G 个生成同时在显存 | 减 G / batch_size / lora_rank |
| **模型崩了**(乱码) | KL 没约束住 | 增大 β,降学习率 |

---

## 📈 11.4.9 评估 GRPO 模型

```python
# 评估 GRPO 模型
eval_result = rl_tool.run({
    "action": "evaluate",
    "model_path": "./models/grpo_full",
    "max_samples": 100,
    "use_lora": True,
})

# 典型结果:
# SFT:    准确率 ~45%
# GRPO:   准确率 ~65%   (+20%!)
```

→ **GRPO 通常带来 10-20% 的提升**。

---

## 🌟 11.4.10 GRPO 的更广应用

### 应用 1:**DeepSeek-R1 风格的推理模型**

```
GRPO + 数学/代码奖励 = R1 风格推理模型
   ↓
具备:
- 长 CoT 推理
- 自我反思
- 错误纠正
```

### 应用 2:**工具使用 Agent**

```
奖励 = 工具调用成功率 + 任务完成度
GRPO 训 → Agent 学会高效用工具
```

### 应用 3:**多模态 Agent**

```
奖励 = 视觉理解准确率
GRPO 训 → 视觉推理 Agent
```

→ **GRPO 是 Agentic-RL 的"通用钥匙"**。

---

## ⚠️ 小白避坑

1. **必须从 SFT 模型开始**
   - 直接 GRPO 预训练模型 → **几乎肯定崩**
2. **`num_generations` 不要乱调**
   - 太小 → 探索不够
   - 太大 → 显存爆炸
   - **8 是黄金值**
3. **学习率要小**
   - SFT 用 5e-5,GRPO 必须 5e-6 或更小
4. **奖励函数有 bug → 训练崩**
   - **先单独测试奖励函数**
5. **小模型期望别太高**
   - 0.6B 模型 GRPO 提升 10-20% 算正常
   - **要看 R1 那种"啊哈时刻"得 7B+**

---

## 📌 11.4 节要点

| 知识点 | 一句话 |
|---|---|
| **GRPO 核心** | 用组内相对奖励代替 Value Model |
| **vs PPO** | **2 模型 vs 4 模型**,更简单更稳 |
| **关键约束** | `batch_size % num_generations == 0` |
| **学习率** | **5e-6**(比 SFT 小 10 倍) |
| **`num_generations`** | 8 是经典值 |
| **预期效果** | SFT 45% → **GRPO 65%** |
| **应用** | DeepSeek-R1 风格推理 / 工具 Agent / 多模态 |

---

## 🔗 延伸阅读

- 上一节:[03-SFT训练实战](03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)
- 下一节:[05-端到端流程与小结](05-%E7%AB%AF%E5%88%B0%E7%AB%AF%E6%B5%81%E7%A8%8B%E4%B8%8E%E5%B0%8F%E7%BB%93.md)
- 论文:DeepSeek-R1 (2025)
- PPO 详解:[Happy-LLM RLHF](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/07-RLHF%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.md)

---

⬅ [03-SFT训练实战](03-SFT%E8%AE%AD%E7%BB%83%E5%AE%9E%E6%88%98.md)	|	➡ [05-端到端流程与小结](05-%E7%AB%AF%E5%88%B0%E7%AB%AF%E6%B5%81%E7%A8%8B%E4%B8%8E%E5%B0%8F%E7%BB%93.md)
