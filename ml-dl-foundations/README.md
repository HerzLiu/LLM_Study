---
tags: [ml-dl-foundations, README, 补完, 前置课]
created: 2026-06-05
source: "用户反向补完需求：已读完 Happy-LLM/Hello-Agents/agentic-rl-knowledge-base，但缺 ML/DL 基础"
---

# ml-dl-foundations · 机器学习 / 深度学习补完计划

> 这套笔记是为「**已经在用 LLM/Agent，但 ML/DL 基础悬空**」的人写的**反向补完课**。
> 不是从大一基础课讲起，而是把你已经在用的 loss / 梯度 / optimizer / LoRA / KL penalty / GRPO 的"底下那一层"补上。

---

## 🎬 一句话定位

| 维度 | 内容 |
|---|---|
| **它是什么** | 把你已经在用但没系统学过的 ML/DL 概念**反向补完**的笔记书 |
| **它不是什么** | 不是 ML 教科书，不是 DL 入门 MOOC，不是数学复习课 |
| **谁适合读** | 已经读完 Happy-LLM / Hello-Agents，能跑 SFT 但不知道 loss 为什么要下降的人 |
| **核心目的** | 让你看 LLM/Agent/RL 代码和论文时，**底下那一层不再悬空** |

---

## 📑 知识库结构（48 篇）

```
ml-dl-foundations/
├── README.md (本文件)
├── 00-总览.md ⭐ 推荐第一篇读
│
├── 01-第1章 机器学习心智模型     (6 节：AI/ML/DL 关系 + 三大范式 + 训练集 + 过拟合 + 评估)
├── 02-第2章 数学最小必要基础     (5 节：线代4件套 + 微积分3件套 + 概率5件套)
├── 03-第3章 经典机器学习入门     (6 节：线性回归 → 逻辑回归 → 决策树速览 → SVM速览 → 转折)
├── 04-第4章 神经网络与深度学习核心 ⭐⭐⭐ (10 节：神经元 → MLP → 激活 → 损失 → 反向传播 → 优化器 → ...)
├── 05-第5章 典型架构串讲         (5 节：CNN 速览 + RNN + Transformer 回顾 + 时间轴)
├── 06-第6章 训练实践与工程       (6 节：超参 + AMP + 分布式 + 梯度问题 + 诊断)
├── 07-第7章 桥接到你已学的世界   ⭐ 收官 (6 节：MLP→Transformer / CE→next-token / SGD→PPO / 损失→RM/DPO/RLVR)
│
└── 附录 (4 篇：术语去重 / PyTorch API / 外部资源 / 是否需要这本书)
```

---

## 🚪 三条阅读入口

| 你的状态 | 入口 | 时间 |
|---|---|---|
| 完全没读过 → | [00-总览](00-%E6%80%BB%E8%A7%88.md) | 15 分钟看完路径选择 |
| 急用（5 天） → | [00-总览](00-%E6%80%BB%E8%A7%88.md) § 🅒 极简版：1 章心智 + 2 章数学 + 4 章 §01-09 + 7 章 §3 | 5 天 |
| 系统学（4-6 周） → | 按 1→2→3→4→5→6→7 顺序读 | 4-6 周 |

---

## 🌉 与已有知识的桥接

这套笔记**只单向跨链出去**（不改你已有的 Happy-LLM / agentic-rl 笔记），但会大量回链：

- **Happy-LLM 第 2 章 Transformer**：补完后再回看会"瞬间懂"FFN/MLP/Attention 的为什么
- **Happy-LLM 第 4-6 章 LLM 训练**：补完后能读懂 `optimizer=AdamW(model.parameters(), lr=5e-5)` 每个词
- **agentic-rl-knowledge-base**：补完后能真正理解 PPO/GRPO 公式（之前可能只看懂表面）
- **附录 04 公式速查表**：补完后那张表的每个公式你都能从头推出来

→ 详细映射见 [00-章节总览](07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## ⚠️ 阅读前请知

1. **这套笔记假设你已经会用 LLM/Agent**：很多例子直接引用 next-token loss / PPO clip / GRPO advantage。不是从"什么是 Python"讲起。
2. **数学只补"看 LLM/RL 论文必须"的部分**：不补 PCA、不补 SVM 完整推导、不补流形学习。
3. **代码片段是概念示意**：标"未严格验证可运行"。要跑真代码请回 D2L 原书或 Happy-LLM 第 5 章。
4. **CV 只在 §5.1 速览**：你目标不是 CV，但需要知道 2012 AlexNet 引爆 DL 的史观。

---

## 🔗 外部参考资源

| 资源 | 用途 | 链接 |
|---|---|---|
| 《动手学深度学习》D2L 中文版 | **主线代码教材** | https://zh.d2l.ai |
| 3Blue1Brown 神经网络系列 | 可视化最强 | https://www.3blue1brown.com/topics/neural-networks |
| 吴恩达 Deep Learning Specialization | 系统视频课 | https://www.deeplearning.ai/courses/deep-learning-specialization/ |
| 李宏毅机器学习 | 中文最好讲解 | https://speech.ee.ntu.edu.tw/~hylee/ml/ |
| 周志华《机器学习》（西瓜书） | 经典 ML 参考 | （纸书） |

→ 完整资源列表见 [03-资源外部链接表](%E9%99%84%E5%BD%95/03-%E8%B5%84%E6%BA%90%E5%A4%96%E9%83%A8%E9%93%BE%E6%8E%A5%E8%A1%A8.md)

---

## 📚 配套已有笔记书

| 笔记书 | 入口 | 角色 |
|---|---|---|
| Happy-LLM | [00-总览](../Happy-LLM/00-%E6%80%BB%E8%A7%88.md) | LLM 怎么造（Transformer / 预训练 / RLHF） |
| Hello-Agents | [00-总览](../Hello-Agents/00-%E6%80%BB%E8%A7%88.md) | Agent 怎么搭 |
| Claude-Code-Source | [00-总览](../Claude-Code-Source/00-%E6%80%BB%E8%A7%88.md) | 工业级 Agent 长啥样 |
| agentic-rl-knowledge-base | [00-总览](../agentic-rl-knowledge-base/00-%E6%80%BB%E8%A7%88.md) | RL / Agentic RL |
| **ml-dl-foundations（本书）** | [00-总览](00-%E6%80%BB%E8%A7%88.md) | **以上 4 本书的"前置课"** |

---

➡ 现在开始：[00-总览](00-%E6%80%BB%E8%A7%88.md)
