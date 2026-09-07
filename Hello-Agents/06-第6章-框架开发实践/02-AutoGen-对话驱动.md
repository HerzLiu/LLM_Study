---
tags: [Hello-Agents, 第6章, AutoGen, 微软, 对话驱动]
chapter: 6
section: 6.2
---

# 6.2 AutoGen —— 对话驱动协作

⬅ [01-为什么需要框架](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%A1%86%E6%9E%B6.md)	|	➡ [03-AgentScope-消息驱动](03-AgentScope-%E6%B6%88%E6%81%AF%E9%A9%B1%E5%8A%A8.md)

> **微软**出品,**对话驱动协作**的多智能体框架。
> **0.7.4** 版本是大重构,从继承式 → **组合式架构**。

---

## 🎬 故事比喻:Slack 工作群

想象一个 Slack 群里:
- 产品经理 PM 提需求
- 工程师 Eng 写代码
- 测试员 QA 审查
- 你(用户)发起任务 + 验收

→ **AutoGen 把这个工作群"自动化"了**:
- 每个角色是个 LLM Agent
- 群规定了发言顺序
- 任务通过**对话**完成

---

## 🎯 6.2.1 AutoGen 的核心机制(0.7.4 新架构)

### 🔄 架构演进

```
旧版 (< 0.5):  类继承设计 → 灵活度不够
   ↓
新版 (0.7.4+): 组合式架构 + 异步优先
```

### 🏗 分层设计

```
┌─────────────────────────────────────┐
│  autogen-agentchat                  │  ← 高层对话接口
│  (开发者用)                          │
└─────────────────────────────────────┘
              ↑ 基于
┌─────────────────────────────────────┐
│  autogen-core                       │  ← 底层基础
│  (LLM 调用 / 消息传递)              │
└─────────────────────────────────────┘
```

### ⚡ 异步优先

```python
# 一切都是 async/await
async def run_team():
    result = await team.run_stream(task=...)
```

**好处**:多 Agent 等 LLM 响应时,不阻塞 → **真正并发**。

---

## 🧱 6.2.2 核心智能体组件

### 🤖 AssistantAgent(助理智能体)

> **任务的主要解决者**,封装 LLM。

```python
from autogen_agentchat.agents import AssistantAgent

product_manager = AssistantAgent(
    name="ProductManager",
    model_client=model_client,
    system_message="你是产品经理...",
)
```

→ **通过不同的 system_message,赋予不同的"专家"角色**。

### 👤 UserProxyAgent(用户代理)

> **双重角色**:
> - **代言人**:发起任务、传达意图
> - **执行器**:可配置执行代码 / 调用工具

```python
from autogen_agentchat.agents import UserProxyAgent

user_proxy = UserProxyAgent(
    name="UserProxy",
    description="代表用户,验证最终代码,完成测试后回复 TERMINATE",
)
```

→ 这种设计**区分了"思考"(Assistant)和"行动"(UserProxy)**。

### 🏛 Team / GroupChat(团队 / 群聊)

> 协调多 Agent 协作的机制。

**轮询群聊(RoundRobinGroupChat)**:
- 智能体按**预定义顺序**依次发言
- 适合**流程固定**的任务

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

team = RoundRobinGroupChat(
    participants=[pm, engineer, reviewer, user_proxy],
    termination_condition=TextMentionTermination("TERMINATE"),
    max_turns=20,
)
```

---

## 🛠 6.2.3 实战:模拟软件开发团队

### 业务目标

**开发一个"实时显示比特币价格"的 Streamlit Web 应用**。

涵盖**完整软件开发流程**:
- 需求分析 → 技术选型 → 编码 → 审查 → 测试

### 团队角色设计

| 角色 | 职责 | 系统消息要点 |
|---|---|---|
| **ProductManager** | 需求 → 开发计划 | 5 个分析维度 + "请工程师开始实现" |
| **Engineer** | 写代码 | Python/Streamlit 专长 + "请代码审查员检查" |
| **CodeReviewer** | 代码审查 | 质量/安全/最佳实践 + "代码审查完成,请用户代理测试" |
| **UserProxy** | 验收 | 完成后回复 TERMINATE |

### 关键代码

#### Step 1:**模型客户端**

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient

def create_openai_model_client():
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
```

→ 兼容 **OpenAI / Azure / Ollama / DeepSeek** 等所有 OpenAI 兼容服务。

#### Step 2:**产品经理智能体**(展示 prompt 设计)

```python
def create_product_manager(model_client):
    system_message = """你是一位经验丰富的产品经理...

核心职责:
1. **需求分析**: 深入理解用户需求
2. **技术规划**: 制定清晰的实现路径
3. **风险评估**: 识别风险
4. **协调沟通**: 与工程师沟通

收到任务时,请按以下结构分析:
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应,分析完成后说"请工程师开始实现"。"""

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )
```

→ ⭐ **关键设计**:**结尾"请 XX 开始" 是流程接力的暗号**。

#### Step 3:**异步主流程**

```python
async def run_software_development_team():
    # ... 初始化 Agent ...

    task = """开发一个比特币价格显示应用:
- 实时显示价格(USD)
- 24 小时趋势
- 用 Streamlit 框架
- 简洁美观,有错误处理"""

    # 流式输出对话过程
    result = await Console(team_chat.run_stream(task=task))
    return result

if __name__ == "__main__":
    asyncio.run(run_software_development_team())
```

→ **`asyncio.run` 是 Python 异步入口**。

---

## 🎮 6.2.4 真实运行流程

```
🚀 启动 AutoGen 软件开发团队协作...

---------- TextMessage (user) ----------
开发一个比特币价格显示应用...

---------- TextMessage (ProductManager) ----------
### 1. 需求理解与分析
...
请工程师开始实现。

---------- TextMessage (Engineer) ----------
### 技术方案实施
[完整 Streamlit 代码]
请代码审查员检查。

---------- TextMessage (CodeReviewer) ----------
### 代码审查
[审查意见 + 改进建议]
代码审查完成,请用户代理测试。

---------- TextMessage (UserProxy) ----------
已经完成需求

---------- TextMessage (UserProxy) ----------
TERMINATE
============================================================
✅ 团队协作完成!
```

→ **整个流程像真实软件团队的协作**,**完全自动化**。

---

## ⭐ 6.2.5 AutoGen 的优势与局限

### ✅ 优势

| 优势 | 解释 |
|---|---|
| **降低建模门槛** | 把复杂协作映射为对话,不用设计状态机 |
| **角色专业化** | 通过 system_message 高度定制 |
| **角色复用** | 一个 Agent 可在多项目用 |
| **流程可预测** | RoundRobinGroupChat 顺序明确 |
| **Human-in-the-loop** | UserProxy 天然支持 |

### ⚠️ 局限

| 局限 | 解释 |
|---|---|
| **不确定性** | LLM 可能产生偏离预期的回复 |
| **"对话式调试"难** | 不是错误堆栈,而是一长串对话历史 |
| **流程僵化** | RoundRobin 太顺序,**不适合非线性任务** |
| **Token 成本高** | 多 Agent 每轮都调 LLM,**贵** |

---

## 🔧 6.2.6 配置非 OpenAI 模型(实战补丁)

用 DeepSeek / 通义千问等?需要传 `model_info`:

```python
model_client = OpenAIChatCompletionClient(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model_info={
        "function_calling": True,
        "max_tokens": 4096,
        "context_length": 32768,
        "vision": False,
        "json_output": True,
        "family": "deepseek",
        "structured_output": True,
    }
)
```

→ 告诉 AutoGen **模型的能力边界**。

---

## ⚠️ 小白避坑

1. **新版与旧版 API 不兼容**
   - 0.7+ 完全重构
   - 看老教程会一头雾水
2. **对话末尾的"接力暗号"必须明确**
   - 不写 → 流程乱掉
   - 用 `"请 XX 开始"` / `"TERMINATE"`
3. **`max_turns` 必设**
   - 防止无限轮询
4. **打印对话流要用 Console + run_stream**
   - 否则看不到实时输出

---

## 🎯 6.2.7 何时选 AutoGen?

```
✅ 适合:
- 流程清晰的多 Agent 协作(软件开发/写报告/数据分析)
- 想模拟真实团队工作的场景
- 需要 Human-in-the-loop
- 微软生态(Azure / Windows 用户)

❌ 不适合:
- 复杂控制流(用 LangGraph)
- 高并发生产系统(用 AgentScope)
- 单 Agent 简单任务(过度工程)
```

---

## 📌 6.2 节要点

| 知识点 | 一句话 |
|---|---|
| **核心哲学** | 对话驱动协作(像 Slack 工作群) |
| **0.7.4 架构** | 组合式 + 异步优先 + 分层 |
| **3 大组件** | AssistantAgent + UserProxyAgent + Team |
| **RoundRobinGroupChat** | 顺序发言,流程化任务首选 |
| **接力暗号** | system_message 中明示下一步给谁 |
| **典型场景** | 软件开发团队 / 文档协作 |
| **致命短板** | 对话式调试 + 流程僵化 |

---

## 🔗 延伸阅读

- 上一节:[01-为什么需要框架](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%A1%86%E6%9E%B6.md)
- 下一节:[03-AgentScope-消息驱动](03-AgentScope-%E6%B6%88%E6%81%AF%E9%A9%B1%E5%8A%A8.md) —— 工程化路线
- 官方:https://microsoft.github.io/autogen/
- 类比:[第 4 章 ReAct](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/02-ReAct%E8%8C%83%E5%BC%8F.md) 的多 Agent 版

---

⬅ [01-为什么需要框架](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E6%A1%86%E6%9E%B6.md)	|	➡ [03-AgentScope-消息驱动](03-AgentScope-%E6%B6%88%E6%81%AF%E9%A9%B1%E5%8A%A8.md)
