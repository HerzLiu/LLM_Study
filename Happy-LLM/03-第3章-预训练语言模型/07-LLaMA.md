---
tags: [Happy-LLM, 第3章, PLM, LLaMA, Decoder-only, 开源LLM]
chapter: 3
section: 3.3.2
---

# 3.3.2 LLaMA —— 开源 LLM 之王

⬅ [06-GPT](06-GPT.md)	|	➡ [08-GLM](08-GLM.md)

> 📅 Meta（Facebook）2023.02 发布 LLaMA-1，至今已有 LLaMA-3（2024.04）
> 🎯 核心信息：**Meta 开源了堪比 GPT-3.5 的模型**，引爆开源 LLM 生态

---

## 🎬 故事比喻：开源版 GPT，把饭桌端到了大家面前

[GPT](06-GPT.md) 是 OpenAI 的私房菜（API 收费 + 闭源）。
**LLaMA = Meta 开源了一桌饭** —— 模型权重免费下载，谁都能拿来微调、部署、商用（LLaMA-2 起）。

→ 直接催生了**国产 LLM 大爆发**：
- **Alpaca / Vicuna / WizardLM**（基于 LLaMA 微调）
- **Qwen / Baichuan / Yi / Skywork**（受 LLaMA 架构启发）
- **LLaMA 已经成为开源 LLM 的"事实标准"**

---

## 🔧 模型架构：Decoder-Only（GPT 流派）

LLaMA 的架构**和 [GPT](06-GPT.md) 几乎一致**，但有几个关键改进：

| 组件 | GPT | **LLaMA** |
|---|---|---|
| 架构类型 | Decoder-only | Decoder-only ✅ |
| 位置编码 | [sin/cos 绝对编码](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/09-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.md) | **RoPE（旋转位置编码）** |
| LayerNorm | LayerNorm | **RMSNorm** |
| 激活函数 | GELU | **SwiGLU** |
| Norm 位置 | Pre-Norm | **Pre-Norm** ✅ |
| 注意力 | 标准多头 | LLaMA-2+ 用 **GQA**（分组查询注意力） |

→ 这些改动看起来小，但**全是经过大量实验验证**的现代 LLM 最佳实践。

### 改进 1：RoPE（旋转位置编码）

- 把位置信息**融入 Q 和 K 的旋转**里
- 既有绝对位置又能反映相对位置
- 支持**长序列外推**
- → 第 5 章会详细讲并手撕代码

### 改进 2：RMSNorm

参考 [RMSNorm 详解](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/07-LayerNorm%E4%B8%8EBatchNorm.md#-%E8%BF%9B%E9%98%B6rmsnormllama-%E7%94%A8%E7%9A%84%E7%AE%80%E5%8C%96%E7%89%88)。
- 速度比 LayerNorm 快 ~10%
- 大规模训练时省时间 = 省钱

### 改进 3：SwiGLU 激活函数

GPT 用 GELU，LLaMA 用 SwiGLU（Swish + GLU 组合）：
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$
- 比 GELU 略复杂但效果更好
- 现代 LLM 普遍采用

### 改进 4：GQA（Grouped-Query Attention，LLaMA-2 起）

> 后面 [横向对比](09-%E6%A8%AA%E5%90%91%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md) 会详细讨论

简单理解：
- **MHA**（标准多头）：每个头独立 Q/K/V → 高质量但 KV-Cache 大
- **MQA**（多查询）：所有头共享 K/V → KV-Cache 极小但效果差
- **GQA**（折中）：每 g 个头共享一组 K/V → **质量和速度的平衡**

---

## 📈 LLaMA 系列演进

### LLaMA-1（2023.02）

| 项 | 数值 |
|---|---|
| **参数版本** | 7B / 13B / 30B / **65B** |
| 预训练 token | 1T |
| 上下文长度 | 2K |
| 训练硬件 | 65B 用 2048 张 A100（80G） × 21 天 |
| **重要意义** | **首个性能堪比 GPT-3 的开源模型**，引爆开源 LLM 生态 |

### LLaMA-2（2023.07）

| 项 | 改进 |
|---|---|
| 参数版本 | 7B / 13B / 34B（未开源）/ **70B** |
| 预训练 token | **2T**（翻倍） |
| 上下文长度 | **4K**（翻倍） |
| 引入技术 | **GQA**（Grouped-Query Attention） |
| **重大变化** | **允许商用**（开源协议宽松化） → 直接引爆产业落地 |

### LLaMA-3（2024.04）

| 项 | 改进 |
|---|---|
| 参数版本 | 8B / **70B** / 400B（训练中） |
| 预训练 token | **15T**（7× LLaMA-2） |
| 上下文长度 | **8K** |
| Tokenizer 词表 | **128K**（远大于 LLaMA-2 的 32K） |
| **重大变化** | 多语言能力大幅提升、tokenizer 效率高 |

### 数据规模演进

```
LLaMA-1: 1T   tokens
LLaMA-2: 2T   tokens  ←  2×
LLaMA-3: 15T  tokens  ←  7.5×
```

→ 印证了 **scaling law**：更多数据 = 更强模型。

---

## 🌳 LLaMA 衍生模型生态

LLaMA 开源后，衍生出庞大的家族：

```
LLaMA（基座）
   │
   ├── Alpaca       ← 斯坦福 SFT 微调
   ├── Vicuna       ← UC 伯克利对话微调
   ├── WizardLM     ← 微软指令微调
   ├── Code Llama   ← Meta 自己的代码版本
   ├── Llama2-Chat  ← Meta 自己的对话版
   │
   └── 影响中文 LLM 架构：
       ├── Qwen（阿里）
       ├── Baichuan（百川）
       ├── Yi（零一万物）
       └── Skywork（昆仑万维）
```

→ 几乎**所有现代开源 LLM 的架构都借鉴或基于 LLaMA**。

---

## ⭐ 加深理解：LLaMA 为什么这么强？

1. **架构上的现代化最佳实践组合**
   - RMSNorm + Pre-Norm + RoPE + SwiGLU + GQA
   - 每一个都是经过大量实验验证的"标配"
2. **数据质量 > 数据数量**
   - LLaMA-1 用 1T token 干翻 GPT-3 175B（用了 300B token）
   - 说明 **更优质的数据 + 更优的架构** 可以超过单纯堆参数
3. **开源策略**
   - 让整个社区帮你做实验、微调、找 bug
   - 形成**生态飞轮**
4. **持续迭代**
   - 1→2→3 都有明显改进
   - 每代都吸收社区反馈

---

## 🆚 LLaMA vs GPT 对比

| 维度 | GPT 系列 | **LLaMA 系列** |
|---|---|---|
| 开源 | ❌（GPT-2 部分开源） | ✅ |
| 商业使用 | API 收费 | LLaMA-2+ 允许商用 |
| 架构创新 | 较少（专注规模化） | RMSNorm、RoPE、GQA、SwiGLU |
| 参数规模 | 最大 175B（GPT-3） | 最大 405B（LLaMA-3） |
| 微调难度 | 难（不开源） | 易（社区工具丰富） |
| 部署 | 只能用 API | 可本地部署 |
| 影响力 | 商业最强 | **开源最强 + 学术研究首选** |

---

## ⚠️ 小白避坑

1. **LLaMA-1 不能商用！** LLaMA-2 起才允许
2. **LLaMA 不是中文模型** —— 原生中文能力弱（训练数据以英文为主）
   - 中文用 Qwen / ChatGLM / DeepSeek 更好
3. **不要混淆 LLaMA 和 Alpaca / Vicuna**：
   - LLaMA = 基座模型（预训练）
   - Alpaca / Vicuna = 在 LLaMA 上微调的对话模型
4. **LLaMA-3 的 tokenizer 跨语言效率高** —— 词表 128K 大很多，但对中文等非英文更友好

---

## 🔗 延伸阅读

- 上一节：[06-GPT](06-GPT.md) —— LLaMA 的"祖宗"
- 下一节：[08-GLM](08-GLM.md) —— 中文开源 LLM 先锋
- 架构基础：[第 2 章 Transformer](../02-%E7%AC%AC2%E7%AB%A0-Transformer%E6%9E%B6%E6%9E%84/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 实战篇：[第 5 章：手撕 LLaMA2](../05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

⬅ [06-GPT](06-GPT.md)	|	➡ [08-GLM](08-GLM.md)
