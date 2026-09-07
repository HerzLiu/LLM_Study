---
tags: [Hello-Agents, 第12章, BFCL, 工具调用, AST匹配]
chapter: 12
section: 12.1-12.2
---

# 12.1-12.2 BFCL:工具调用评估基准

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-GAIA通用评估](02-GAIA%E9%80%9A%E7%94%A8%E8%AF%84%E4%BC%B0.md)

> **BFCL**(Berkeley Function Calling Leaderboard)—— **工具调用评估的金标准**。

---

## 🎬 故事比喻:**驾照路考**

| | 真实驾照 | **BFCL** |
|---|---|---|
| 考啥? | 开车技术 | **Agent 用工具的技术** |
| 评什么? | 倒车/侧方位/超车 | **single/multiple/parallel/irrelevance** |
| 评分? | 扣分制 | **AST 匹配准确率** |
| 通过? | 上路 | **上榜单** |

→ **BFCL 是 Agent 工具调用能力的"驾照"**。

---

## ❓ 12.1 为什么需要评估?

### Agent 开发的"最后一公里"

```
没有评估 → 自己感觉良好(其实拉胯)
有评估   → 知道差距 → 针对优化 → 持续提升
```

### Agent 评估的 3 大维度

| 维度 | 说明 |
|---|---|
| **能力评估** | Agent 能干啥(BFCL/GAIA) |
| **效率评估** | 多少 token / 多少时间 |
| **安全评估** | 是否有害、是否对齐价值观 |

→ 本章重点讲**能力评估**。

---

## 🏆 12.2.1 BFCL 介绍

> **BFCL**(Berkeley Function Calling Leaderboard)
> 由 UC Berkeley 提出,**工具调用评估的金标准**。

### 4 大评估类别

| 类别 | 测试内容 | 典型样本 |
|---|---|---|
| **simple** | 单个函数调用 | "今天北京天气?" → `get_weather("Beijing")` |
| **multiple** | 多个函数中选对的 | 给 5 个函数,选**正确那个** |
| **parallel** | 并行调用多个 | 同时查 3 个城市天气 |
| **irrelevance** | **判断是否需要工具** | "讲个笑话" → **不调工具** |

→ 4 大场景**覆盖工具调用的全部典型情况**。

---

## 📦 12.2.2 BFCL 数据集结构

```json
{
  "id": "simple_001",
  "question": "What's the weather like in Beijing today?",
  "function": [
    {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "The city name"}
        },
        "required": ["location"]
      }
    }
  ],
  "ground_truth": [
    {"name": "get_weather", "arguments": {"location": "Beijing"}}
  ]
}
```

| 字段 | 含义 |
|---|---|
| `question` | 用户的自然语言请求 |
| `function` | 可用函数列表(签名 + 描述) |
| `ground_truth` | **标准答案**(期望的函数调用) |

---

## 🧠 12.2.3 核心评估算法:**AST 匹配**

> **AST**(Abstract Syntax Tree)—— **抽象语法树匹配**
> 比字符串匹配**智能得多**。

### 数学定义

$$\text{AST\_Match}(\hat{f}, f^*) = \begin{cases} 1 & \text{if } \text{parse}(\hat{f}) \equiv \text{parse}(f^*) \\ 0 & \text{otherwise} \end{cases}$$

其中:
- `parse(·)`:把函数调用解析为**抽象语法树**
- `≡`:语法树**等价**

### AST 等价的 3 条规则

| 规则 | 解释 |
|---|---|
| **函数名精确匹配** | `get_weather` ≠ `get_temperature` |
| **参数键值对集合相等** | **顺序无关** |
| **每个参数值在语义上等价** | `2+3` ≡ `5` |

### AST 匹配示例

#### ✅ 匹配成功

```
预测: get_weather(city="Beijing", unit="celsius")
标准: get_weather(unit="celsius", city="Beijing")
       ↑ 顺序不同也能匹配
```

```
预测: calculate(x=2+3)
标准: calculate(x=5)
       ↑ 等价表达式也能匹配
```

#### ❌ 匹配失败

```
预测: get_temperature(city="Beijing")   ← 函数名错
标准: get_weather(city="Beijing")
```

```
预测: get_weather(city="Shanghai")       ← 参数值错
标准: get_weather(city="Beijing")
```

→ AST 匹配**抓本质,不被表面差异骗**。

---

## 📊 12.2.4 BFCL 五大评估指标

### 指标 1:**准确率(Accuracy)** ⭐ 最核心

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \text{AST\_Match}(\hat{f}_i, f^*_i)$$

| Accuracy | 含义 |
|---|---|
| 1.0 | **全对** |
| 0.8 | 80% 正确 |
| 0.0 | 全错 |

### 指标 2:**AST 匹配率**

```
与准确率相同,强调 AST 匹配
```

### 指标 3:**分类准确率**

$$\text{Accuracy}_c = \frac{1}{N_c} \sum_{i \in \mathcal{D}_c} \text{AST\_Match}(\hat{f}_i, f^*_i)$$

→ 看**每个类别**的表现。

### 指标 4:**加权准确率**

$$\text{Weighted Accuracy} = \sum_{c} w_c \cdot \text{Accuracy}_c$$

→ 不同类别**难度不同**,加权综合。

### 指标 5:**错误率**

$$\text{Error Rate} = 1 - \text{Accuracy}$$

### 示例分析

```
simple:        准确率 95%
multiple:      准确率 82%
parallel:      准确率 68%
irrelevance:   准确率 75%

加权(权重相等):  (95 + 82 + 68 + 75) / 4 = 80%

洞察:
- simple 强 → 基础工具调用 OK
- parallel 弱 → 并行场景需提升
```

---

## 📥 12.2.5 获取 BFCL 数据集

### 方法 1:**官方仓库克隆**(推荐)

```bash
git clone https://github.com/ShishirPatil/gorilla.git temp_gorilla
cd temp_gorilla/berkeley-function-call-leaderboard

# 查看 v4 数据集
ls bfcl_eval/data/
# → BFCL_v4_simple_python.json
#   BFCL_v4_multiple.json
#   BFCL_v4_parallel.json
#   ...
```

**优势**:
- ✅ 包含**完整的 ground truth**
- ✅ 数据格式与**官方评估工具一致**
- ✅ 支持 **BFCL v4 最新版**

### 方法 2:**HelloAgents 加载**

```python
from hello_agents.evaluation import BFCLDataset

dataset = BFCLDataset(
    bfcl_data_dir="./temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
    category="simple_python",
)
data = dataset.load()
print(f"✅ 加载 {len(data)} 个测试样本")
```

---

## 🚀 12.2.6 HelloAgents BFCL 评估实战

### 方式 1:**BFCLEvaluationTool 一键评估**(推荐)

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import BFCLEvaluationTool

# 1. 创建 Agent
agent = SimpleAgent(name="TestAgent", llm=HelloAgentsLLM())

# 2. 创建评估工具
bfcl_tool = BFCLEvaluationTool()

# 3. 一键评估(自动完成所有步骤)
results = bfcl_tool.run(
    agent=agent,
    category="simple_python",
    max_samples=5,
)

# 4. 看结果
print(f"准确率: {results['overall_accuracy']:.2%}")
print(f"正确: {results['correct_samples']}/{results['total_samples']}")
```

### 自动执行流程(4 步)

```
1. ✅ 加载 BFCL 数据集
2. ✅ 运行 Agent 获取预测
3. ✅ 导出 BFCL 官方格式
4. ✅ 运行 BFCL 官方评估
5. ✅ 生成 Markdown 报告
```

→ **5 个动作**,**完全自动化**。

### 生成的报告示例

```markdown
# BFCL 评估报告

**生成时间**: 2025-10-11 00:59:38

## 📊 评估概览
- 智能体: TestAgent
- 评估类别: simple_python
- 总体准确率: 100.00%
- 正确样本数: 5/5

## 📈 详细指标
### 分类准确率
- simple_python: 100.00% (5/5)

## 📊 准确率可视化
准确率: ████████████████████ 100.00%

## 💡 建议
- ✅ 表现优秀!智能体在工具调用方面表现出色。
```

### 方式 2:**底层 Dataset + Evaluator**(灵活)

```python
from hello_agents.evaluation import BFCLDataset, BFCLEvaluator

# 1. 加载数据集
dataset = BFCLDataset(
    bfcl_data_dir="./temp_gorilla/.../bfcl_eval/data",
    category="simple_python",
)
data = dataset.load()

# 2. 创建评估器
evaluator = BFCLEvaluator(
    dataset=dataset,
    category="simple_python",
    evaluation_mode="ast",   # AST 匹配
)

# 3. 评估
results = evaluator.evaluate(agent, max_samples=10)

# 4. 导出官方格式
evaluator.export_to_bfcl_format(
    results,
    output_path="./evaluation_results/my_results.json",
)
```

---

## 🔬 12.2.7 评估流程图

```
┌────────────────────────────────────┐
│  1. 加载 BFCL 数据集                 │
│     - 测试问题                       │
│     - 可用函数                       │
│     - Ground Truth                  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  2. Agent 生成预测                   │
│     - 接收问题                       │
│     - 选择函数 + 生成参数             │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  3. AST 匹配                        │
│     - 解析预测和标准答案为 AST        │
│     - 比较语法树结构                  │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  4. 计算指标                         │
│     - 准确率 / 分类准确率 / ...      │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  5. 生成报告 + 提交榜单(可选)        │
└────────────────────────────────────┘
```

---

## 🌟 12.2.8 提升 BFCL 分数的方向

> ⚠️ HelloAgents 默认的 SimpleAgent **基础但有提升空间**。

### 方向 1:**用更强的 LLM**
- GPT-4 / Claude 3.5 / Gemini 2.0 等
- **支持原生 Function Calling** 的更佳

### 方向 2:**优化工具调用格式**
- 自定义格式 → **OpenAI Function Calling**
- 详见 [[../07-第7章-构建你的智能体框架/05-工具系统#🔑 3. **`get_openai_schema()` 桥接 Function Calling**|7.5 节]]

### 方向 3:**针对不同类别优化**

| 类别 | 优化策略 |
|---|---|
| **simple** | 优化 prompt 让 LLM 准确理解参数 |
| **multiple** | 强调"先选对函数,再填参数" |
| **parallel** | 教 LLM 识别"可以并行"的场景 |
| **irrelevance** | 训练 LLM **判断何时不需要工具** |

### 方向 4:**渐进式评估策略**

```python
# Step 1: 快速测试(5 样本)
results = bfcl_tool.run(agent, category="simple_python", max_samples=5)

# Step 2: 准确率 > 80% → 中规模(50 样本)
if results['overall_accuracy'] > 0.8:
    results = bfcl_tool.run(agent, category="simple_python", max_samples=50)

# Step 3: 仍 > 80% → 完整评估
if results['overall_accuracy'] > 0.8:
    results = bfcl_tool.run(agent, category="simple_python", max_samples=0)   # 全部
```

→ **节省成本** + **逐步验证**。

---

## 📤 12.2.9 提交到 BFCL 官方排行榜

### 步骤

```
1. 准备材料
   - 模型描述文档
   - 评估结果文件(所有类别)
   - 模型访问方式(API/开源)

2. 提交 Pull Request
   - GitHub: https://github.com/ShishirPatil/gorilla
   - 参考 CONTRIBUTING.md

3. 等待审核
   - BFCL 团队验证
   - 通过后上榜
```

→ **学术界 + 工业界的双重认可**。

---

## ⚠️ 小白避坑

1. **必须用 BFCL v4**
   - 旧版数据格式不同
   - 看官方仓库最新版
2. **`partial-eval` 标志**
   - 小样本评估必须加
   - 否则官方工具报错
3. **`model_name` 一致**
   - 导出 + 评估时**保持一致**
4. **不要拿 SimpleAgent 直接上榜**
   - 是教学用,生产用 OpenAI Function Calling
5. **AST 匹配也有局限**
   - 复杂表达式可能匹配失败
   - 看官方 issue 跟进

---

## 📌 12.1-12.2 节要点

| 知识点 | 一句话 |
|---|---|
| **BFCL 定位** | 工具调用评估**金标准**(UC Berkeley) |
| **4 大类别** | simple / multiple / parallel / irrelevance |
| **AST 匹配** | **抓本质**,不被表面差异骗 |
| **5 大指标** | Accuracy / AST Match / Category / Weighted / Error |
| **HelloAgents 工具** | `BFCLEvaluationTool` 一键评估 |
| **提升方向** | 更强 LLM + Function Calling + 类别针对优化 |
| **上榜** | GitHub PR + 官方审核 |

---

## 🔗 延伸阅读

- 上一节:[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节:[02-GAIA通用评估](02-GAIA%E9%80%9A%E7%94%A8%E8%AF%84%E4%BC%B0.md)
- 工具系统:[05-工具系统](../07-%E7%AC%AC7%E7%AB%A0-%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E6%A1%86%E6%9E%B6/05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)
- 官方:https://gorilla.cs.berkeley.edu/leaderboard.html

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-GAIA通用评估](02-GAIA%E9%80%9A%E7%94%A8%E8%AF%84%E4%BC%B0.md)
