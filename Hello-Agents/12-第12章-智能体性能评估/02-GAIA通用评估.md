---
tags: [Hello-Agents, 第12章, GAIA, 通用AI, 准精确匹配]
chapter: 12
section: 12.3
---

# 12.3 GAIA:通用 AI 助手能力评估

⬅ [01-BFCL工具调用评估](01-BFCL%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E8%AF%84%E4%BC%B0.md)	|	➡ [03-数据生成质量评估](03-%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0.md)

> **GAIA**(General AI Assistants)—— **Meta AI + Hugging Face** 出品的**通用 AI 评估标杆**。

---

## 🎬 故事比喻:**全能助理选拔赛**

| | 普通"工具调用考试" | **GAIA 全能考试** |
|---|---|---|
| 任务 | 单一明确 | **多步推理 + 综合能力** |
| 输入 | 文本 | **文本 + 图 + PDF + Excel** |
| 难度 | 中 | **L1 / L2 / L3 三级** |
| 评什么? | 工具调用 | **多步推理 + 知识 + 多模态 + 浏览 + 文件处理** |

→ GAIA = **AI 助手的"司法考试"**。

---

## 🎯 12.3.1 GAIA 设计理念

### 核心:**真实世界问题需要综合能力**

```
单一能力(BFCL):       Agent 能调对工具
综合能力(GAIA):       Agent 能完成 真实 复杂 多步骤 任务
```

### 5 大综合能力

| 能力 | 例子 |
|---|---|
| **多步推理** | 复杂问题 → 拆解为多个子问题 |
| **知识运用** | 内置知识 + 外部知识库 |
| **多模态理解** | 文本 + 图片 + PDF + 表格 |
| **网页浏览** | 从互联网获取最新信息 |
| **文件操作** | 读取 / 处理 / 分析各种文件 |

→ **像人类助理一样**,**啥都能干**。

---

## 📦 12.3.2 GAIA 数据集结构

### 数据集统计

| 项 | 数据 |
|---|---|
| 总样本 | **466 个**真实问题 |
| 难度级别 | **L1 / L2 / L3** |
| 多模态附件 | 图片 / PDF / CSV / Excel |
| 评估方式 | 准精确匹配 |

### 难度分布

| 级别 | 描述 | 推理步骤 |
|---|---|---|
| **Level 1** | 简单任务 | 0-1 步 |
| **Level 2** | 中等任务 | 2-5 步 |
| **Level 3** | 困难任务 | **6+ 步,可能需复杂工具组合** |

### 样本示例

```json
{
  "task_id": "gaia_001",
  "Question": "What is the total population of the top 3 most populous cities in California?",
  "Level": 2,
  "Final answer": "12847521",
  "file_name": "",
  "Annotator Metadata": {
    "Steps": [
      "Search for most populous cities in California",
      "Get population data for top 3 cities",
      "Sum the populations"
    ],
    "Number of steps": 3,
    "Tools": ["web_search", "calculator"]
  }
}
```

| 字段 | 含义 |
|---|---|
| `Question` | 用户问题 |
| `Level` | 难度(1-3) |
| `Final answer` | 标准答案 |
| `file_name` | 附件(若有) |
| `Annotator Metadata` | 标注者元数据(预期步骤/工具) |

---

## 🧠 12.3.3 评估算法:**准精确匹配**(Quasi Exact Match)

> GAIA 不是 BFCL 的 AST 匹配,**而是文本归一化后比对**。

### 数学定义

$$\text{QEM}(\hat{a}, a^*) = \begin{cases} 1 & \text{if } \text{normalize}(\hat{a}) = \text{normalize}(a^*) \\ 0 & \text{otherwise} \end{cases}$$

### 归一化规则(3 大类型)

#### 类型 1:**数字**

```python
# 规则:
# - 移除逗号分隔符:1,000 → 1000
# - 移除单位符号:$100 → 100, 50% → 50

"$1,234.56" → "1234.56"
```

#### 类型 2:**字符串**

```python
# 规则:
# - 转小写:Apple → apple
# - 移除冠词:the apple → apple
# - 移除多余空格:hello  world → hello world
# - 移除末尾标点:hello. → hello

"The United States of America" → "united states of america"
```

#### 类型 3:**列表**

```python
# 规则:
# - 按逗号分隔
# - 每个元素归一化
# - 按字母排序
# - 重新连接

"Paris, London, Berlin" → "berlin,london,paris"
```

→ **细节抠到位**,**减少误判**。

---

## 📊 12.3.4 GAIA 4 大评估指标

### 指标 1:**精确匹配率**(Exact Match Rate)

$$\text{EMR} = \frac{1}{N} \sum_{i=1}^{N} \text{QEM}(\hat{a}_i, a^*_i)$$

→ **核心指标**,GAIA 排行榜按这个排名。

### 指标 2:**分级准确率**

$$\text{Acc}_L = \frac{1}{N_L} \sum_{i \in \mathcal{D}_L} \text{QEM}(\hat{a}_i, a^*_i)$$

→ 看 **Level 1/2/3 各自的表现**。

### 指标 3:**难度递进下降率**

$$\text{Drop}_{L \to L+1} = \frac{\text{Acc}_L - \text{Acc}_{L+1}}{\text{Acc}_L}$$

→ 衡量 Agent 的**抗复杂性**。

### 指标 4:**平均推理步骤数**

$$\bar{S} = \frac{1}{N_{\text{correct}}} \sum_{i \in \text{correct}} s_i$$

→ 看 Agent **效率**(用最少步数完成)。

### 示例分析

```
样本数: 10

Level 1: 3 个,对了 3 个 → 100%
Level 2: 3 个,对了 2 个 → 67%
Level 3: 4 个,对了 2 个 → 50%

精确匹配率:                70%
Level 1 → 2 下降率:        33%  (1.00 → 0.67)
Level 2 → 3 下降率:        25%  (0.67 → 0.50)

洞察:
- 整体 70% 还行
- L2 下降明显,中等任务薄弱
- L3 50% 在复杂任务上有提升空间
```

---

## 📋 12.3.5 GAIA 官方系统提示词(必须用)

> ⚠️ **GAIA 对答案格式有严格要求**,**必须用官方 prompt**!

```python
GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question.
Report your thoughts, and finish your answer with the following template:
FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma
separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use
units such as $ or percent sign unless specified otherwise.

If you are asked for a string, don't use articles, neither abbreviations (e.g. for
cities), and write the digits in plain text unless specified otherwise.

If you are asked for a comma separated list, apply the above rules depending of
whether the element to be put in the list is a number or a string."""
```

### 关键格式要求

| 要求 | 说明 |
|---|---|
| **必须 `FINAL ANSWER: [答案]`** | 格式硬性 |
| **数字**:不用逗号分隔符 + 不用单位 | `1234.56` 不是 `$1,234.56` |
| **字符串**:不用冠词/缩写 + 数字写全 | `united states` 不是 `the U.S.` |
| **列表**:逗号分隔 + 字母排序 | `berlin,london,paris` |

→ **format 不对,自动算错**。

---

## 📥 12.3.6 获取 GAIA 数据集(需申请)

> ⚠️ GAIA 是 **受限数据集**(Gated Dataset)。

### 步骤 1:**申请访问**

```
1. 访问 https://huggingface.co/datasets/gaia-benchmark/GAIA
2. 点击 "Request access"
3. 填写申请表(几秒内批准)
4. 获取 Token: https://huggingface.co/settings/tokens
```

### 步骤 2:**配置环境变量**

```bash
# .env
HF_TOKEN=hf_your_token_here
```

### 步骤 3:**HelloAgents 自动下载**

```python
from hello_agents.evaluation import GAIADataset
import os

os.environ["HF_TOKEN"] = "hf_your_token_here"

# 首次运行会自动下载到 ./data/gaia/
dataset = GAIADataset(
    dataset_name="gaia-benchmark/GAIA",
    split="validation",   # 或 "test"
    level=1,              # 1/2/3/None(全部)
)
items = dataset.load()
print(f"✅ 加载 {len(items)} 个测试样本")
```

### 数据集目录结构

```
./data/gaia/
├── 2023/
│   ├── validation/
│   │   ├── metadata.jsonl   (165 个问题)
│   │   └── *.png, *.pdf, *.csv  (附件)
│   └── test/
│       ├── metadata.jsonl   (301 个问题)
│       └── ...
└── README.md
```

---

## 🚀 12.3.7 HelloAgents GAIA 评估实战

### 方式 1:**GAIAEvaluationTool 一键评估**(推荐)

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import GAIAEvaluationTool

# GAIA 官方提示词(必须用)
GAIA_SYSTEM_PROMPT = """You are a general AI assistant..."""

# 1. 创建 Agent
agent = SimpleAgent(
    name="TestAgent",
    llm=HelloAgentsLLM(),
    system_prompt=GAIA_SYSTEM_PROMPT,   # ⭐ 关键
)

# 2. 创建评估工具
gaia_tool = GAIAEvaluationTool()

# 3. 一键评估
results = gaia_tool.run(
    agent=agent,
    level=1,              # L1 简单任务
    max_samples=5,
    export_results=True,   # 导出官方格式
    generate_report=True,  # 生成报告
)

# 4. 看结果
print(f"精确匹配率: {results['exact_match_rate']:.2%}")
print(f"部分匹配率: {results['partial_match_rate']:.2%}")
print(f"正确数: {results['exact_matches']}/{results['total_samples']}")
```

### 自动执行流程(3 步)

```
1. ✅ 加载 GAIA 数据集(自动下载)
2. ✅ Agent 运行评估
3. ✅ 导出 GAIA 格式 (JSONL)
4. ✅ 生成提交说明文件
5. ✅ 生成评估报告
```

### 生成的 GAIA 格式结果(JSONL)

```jsonl
{"task_id": "e1fc63a2-...", "model_answer": "24000", "reasoning_trace": "..."}
{"task_id": "8e867cd7-...", "model_answer": "3", "reasoning_trace": "..."}
```

→ **直接可提交官方排行榜**。

### 方式 2:**底层 Dataset + Evaluator**

```python
from hello_agents.evaluation import GAIADataset, GAIAEvaluator

# 1. 加载数据集
dataset = GAIADataset(level=1)
items = dataset.load()

# 2. 创建评估器
evaluator = GAIAEvaluator(dataset=dataset, level=1)

# 3. 评估
results = evaluator.evaluate(agent, max_samples=5)

# 4. 导出
evaluator.export_to_gaia_format(
    results,
    "gaia_results.jsonl",
    include_reasoning=True,
)
```

---

## 📤 12.3.8 提交到 GAIA 官方排行榜

### 步骤

```
1. 用 GAIAEvaluationTool 评估
   → 生成 gaia_level1_result_*.jsonl

2. 访问排行榜:
   https://huggingface.co/spaces/gaia-benchmark/leaderboard

3. 上传 JSONL 文件

4. 填写信息:
   - 模型名称
   - 模型描述
   - 联系方式

5. 等待评测(几分钟到几小时)

6. 上榜!
```

### 提交前检查

```python
import json

with open("evaluation_results/gaia_official/gaia_level1_result_*.jsonl") as f:
    for line in f:
        result = json.loads(line)
        # 验证字段
        assert "task_id" in result
        assert "model_answer" in result
        # 验证 FINAL ANSWER 格式
        # ...
```

---

## ⚠️ 12.3.9 GAIA 评估的现实

### 难度有多大?

| 模型 | Level 1 | Level 2 | Level 3 | 总分 |
|---|---|---|---|---|
| **人类** | 92% | 92% | 92% | **92%** |
| **GPT-4 + 工具** | 30% | 9% | 0% | **15%** |
| **AutoGPT** | 15% | 0% | 0% | **5%** |

→ **GAIA 极难,人类 92% vs AI 15%**。
→ **真正的 AGI 基准之一**。

### 关键差距

```
- 多步推理:AI 在长链推理上易出错
- 多模态融合:AI 跨模态理解弱
- 真实工具调用:AI 难以稳定使用浏览器/Excel
```

---

## 🌟 12.3.10 BFCL vs GAIA 对比

| 维度 | **BFCL** | **GAIA** |
|---|---|---|
| **关注** | 工具调用 | **通用能力** |
| **难度** | 中 | **极高**(L3) |
| **输入** | 文本 | **多模态**(文本+图+文件) |
| **评估算法** | AST 匹配 | **准精确匹配** |
| **样本数** | 数百-数千 | 466 |
| **应用** | 评估工具型 Agent | 评估**全能** Agent |

### 互补关系

```
BFCL:  保证 Agent 会"用工具"
GAIA:  保证 Agent 会"做事"

二者都通过 = Agent 真正强大
```

---

## ⚠️ 小白避坑

1. **不用 GAIA 官方 prompt → 直接 0 分**
   - 格式不符 → 答案被判错
2. **HF_TOKEN 没设 → 下载失败**
   - 必须先在 HuggingFace 申请 access
3. **L1 都做不好就别上 L3**
   - 逐级评估
4. **SimpleAgent 在 GAIA 上很弱**
   - 是正常的(GAIA 太难)
   - 真要刷分用**多 Agent + 工具调用 + 浏览**
5. **提交结果别造假**
   - GAIA 团队会**复核**

---

## 📌 12.3 节要点

| 知识点 | 一句话 |
|---|---|
| **GAIA 定位** | 通用 AI 助手评估**金标准** |
| **3 难度级别** | L1(0-1 步)/ L2(2-5 步)/ L3(6+ 步) |
| **5 大能力** | 推理 / 知识 / 多模态 / 浏览 / 文件 |
| **准精确匹配** | 数字/字符串/列表 **分类归一化** |
| **官方 prompt 必须用** | 格式不符 → 0 分 |
| **HelloAgents 工具** | `GAIAEvaluationTool` 一键评估 |
| **难度** | **GPT-4 + 工具只有 15%**,人类 92% |

---

## 🔗 延伸阅读

- 上一节:[01-BFCL工具调用评估](01-BFCL%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E8%AF%84%E4%BC%B0.md)
- 下一节:[03-数据生成质量评估](03-%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0.md)
- 排行榜:https://huggingface.co/spaces/gaia-benchmark/leaderboard
- 论文:GAIA: A Benchmark for General AI Assistants (Mialon et al., 2023)

---

⬅ [01-BFCL工具调用评估](01-BFCL%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E8%AF%84%E4%BC%B0.md)	|	➡ [03-数据生成质量评估](03-%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0.md)
