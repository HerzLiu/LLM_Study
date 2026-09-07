---
tags: [Hello-Agents, 第10章, MCP, 自定义服务器, 总结]
chapter: 10
section: 10.5-10.6
---

# 10.5-10.6 构建自定义 MCP 服务器 + 章节小结

⬅ [04-ANP协议简介](04-ANP%E5%8D%8F%E8%AE%AE%E7%AE%80%E4%BB%8B.md)	|	➡ [进入第 11 章](../11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## 🎬 故事比喻:从"用别人的轮子"到"造自己的轮子"

10.2 我们用了**社区现成的 MCP 服务器**(GitHub / 文件系统等)。
但真实业务有**独特需求** —— **得自己造**。

→ **本节教你:**写一个**自定义 MCP 服务器**(只需 10 行核心代码)。

---

## 🛠 10.5.1 创建第一个 MCP 服务器(用 FastMCP)

> HelloAgents 基于 **FastMCP** 库,**几行就能造 MCP 服务器**。

### Hello World:计算器服务器

```python
"""my_calculator_server.py"""
from fastmcp import FastMCP

# 创建 MCP 服务器
mcp = FastMCP("Calculator Server")


# ⭐ 用 @mcp.tool() 装饰器定义工具(自动暴露)
@mcp.tool()
def add(a: float, b: float) -> float:
    """加法计算

    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        a + b 的结果
    """
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """乘法计算"""
    return a * b


@mcp.tool()
def greet(name: str) -> str:
    """友好问候"""
    return f"你好, {name}!我是计算器服务器。"


# 启动服务器(Stdio 模式)
if __name__ == "__main__":
    mcp.run()
```

→ **就这么简单**!**3 个工具**,**< 30 行代码**。

### 关键设计

| 元素 | 说明 |
|---|---|
| `FastMCP("Server Name")` | 创建服务器实例 |
| **`@mcp.tool()`** | 装饰器,**自动暴露函数为 MCP 工具** |
| **类型注解** | `def add(a: float, b: float) -> float` → 自动生成 JSON Schema |
| **Docstring** | **变成工具的 description**(LLM 看的!) |
| `mcp.run()` | 启动 Stdio 模式服务器 |

---

## 🧪 10.5.2 测试自定义 MCP 服务器

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

# 连接你自己的 MCP 服务器
my_tool = MCPTool(
    name="my_calc",
    server_command=["python", "my_calculator_server.py"],
)
agent.add_tool(my_tool)
# ✅ 自动展开为 my_calc_add, my_calc_multiply, my_calc_greet

# Agent 调用
response = agent.run("计算 15 加 27")
print(response)   # → 42
```

→ **自定义服务器 → MCPTool 自动展开 → Agent 直接用**。

---

## 🎯 10.5.3 进阶:添加 Resources 和 Prompts

### Resources(被动数据)

```python
@mcp.resource("file:///config/settings.json")
def get_settings() -> str:
    """读取配置文件"""
    with open("settings.json") as f:
        return f.read()


@mcp.resource("data://users/{user_id}")
def get_user(user_id: str) -> dict:
    """根据 ID 查询用户(参数化资源)"""
    return {"id": user_id, "name": f"User-{user_id}"}
```

### Prompts(模板)

```python
@mcp.prompt()
def code_review(language: str = "python") -> str:
    """生成代码审查 prompt 模板"""
    return f"""请审查以下 {language} 代码:

要求:
1. 检查代码风格
2. 找潜在 bug
3. 给出优化建议

代码如下:
{{code}}
"""
```

---

## 🌐 10.5.4 部署:上传 MCP 服务器到社区

> 写好的 MCP 服务器可以**分享给全世界**。

### 上传到 MCP Servers Registry

```bash
# 1. 打包成 npm 包(JavaScript)
# 或 PyPI 包(Python)

# 2. 提交到 https://mcpservers.org/

# 3. 用户即可:
#    npx -y @your-org/your-mcp-server
```

→ **你也可以贡献 MCP 生态**!

### 部署成 HTTP 服务

```python
# 改成 HTTP 模式(便于远程访问)
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8080)
```

→ 然后用户用:
```python
client = MCPClient("http://your-server.com:8080")
```

---

## 📝 10.6 章节小结

### 核心收获

#### 收获 1:**3 大通信协议**

| 协议 | 一句话 | 类比 |
|---|---|---|
| **MCP** | Agent ↔ 工具/数据 | USB-C |
| **A2A** | Agent ↔ Agent | Slack 工作群 |
| **ANP** | Agent ↔ 网络 | DNS + 黄页 |

#### 收获 2:**协议栈思想**

```
ANP (找 Agent)
   ↓
A2A (谈协作)
   ↓
MCP (调工具)
```

#### 收获 3:**MCP vs Function Calling**

| | Function Calling | MCP |
|---|---|---|
| 性质 | **LLM 能力** | **基础设施协议** |
| 关系 | **互补** | |

#### 收获 4:**MCPTool 自动展开**

```
1 个 MCPTool → 服务器的 N 个工具自动暴露
```

#### 收获 5:**社区 MCP 生态**

- mcpservers.org / awesome-mcp-servers
- 数百个现成服务器,**即装即用**

---

## 🌟 10.6.1 设计哲学

### 哲学 1:**标准化是力量**
- USB-C / TCP-IP / MCP 都是同样的思想
- **标准化 → 互操作 → 生态繁荣**

### 哲学 2:**协议 ≠ 框架**
- MCP/A2A/ANP 是协议
- LangChain/AutoGen 是框架
- **二者协同**

### 哲学 3:**渐进式发展**
- MCP 最成熟 → 优先用
- A2A 发展中 → 关注
- ANP 早期 → 等成熟

---

## 🆚 三协议生态成熟度对比

| 协议 | 成熟度 | 学习优先级 | 当下建议 |
|---|---|---|---|
| **MCP** | ⭐⭐⭐⭐⭐ | **最高** | **必学 + 实战** |
| **A2A** | ⭐⭐⭐ | 中 | 了解概念 + 多 Agent 用框架 |
| **ANP** | ⭐⭐ | 低 | 了解未来方向 |

---

## 🎯 10.6.2 实战选型决策

```
你想干啥?
   │
   ├── Agent 用外部工具?
   │   └──→ ⭐ MCP (优先)
   │
   ├── 多 Agent 协作?
   │   ├──→ 学习 → A2A 了解思想
   │   └──→ 生产 → AutoGen/AgentScope/CrewAI
   │
   ├── 大规模 Agent 网络?
   │   └──→ 关注 ANP 演进 + 用 MCP 工具市场
   │
   └── 想造自己的工具?
       └──→ ⭐ 写 MCP 服务器(本节)
```

---

## ⚠️ 整章踩坑清单

| 坑 | 解法 |
|---|---|
| 觉得 MCP 是 LangChain 竞品 | 它是协议,不是框架 |
| 写 MCP 工具没写 docstring | LLM 看不懂工具用途 |
| 多 MCP 服务器没加 `name` 前缀 | 工具名冲突 |
| 用 A2A 替代多 Agent 框架 | 协议不成熟,用框架 |
| 等 ANP 成熟才做大规模 | 当下用 MCP 工具市场 |
| MCP 自定义服务器没加类型注解 | JSON Schema 生成失败 |

---

## ❓ 本章自测题

→ 见 [自测题汇总-第 10 章](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#%E7%AC%AC-10-%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE)

---

## 📌 本章一句话总结

**协议是 Agent 圈的基础设施层**:
- **MCP** 让 Agent 接入万物
- **A2A** 让 Agent 互相协作
- **ANP** 让 Agent 组建网络

**学透 MCP 就掌握了当下 80% 的 Agent 互联能力**。

---

## 🚀 下一章预告

**第 11 章 Agentic-RL** —— 用强化学习训练 Agent:
- 从 LLM 训练 → Agentic RL 的演进
- **SFT + GRPO** 训练实战(类似 DeepSeek-R1 的训练方法)
- 数学推理 Agent 训练实战
- 端到端训练流程 + 分布式 + 生产部署

→ 这是**让 Agent 真正"自主进化"**的关键。

---

## 🔗 延伸阅读

- 上一节:[04-ANP协议简介](04-ANP%E5%8D%8F%E8%AE%AE%E7%AE%80%E4%BB%8B.md)
- **下一章**:[00-章节总览](../11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- HelloAgents 工具:[05-工具系统](../07-%E7%AC%AC7%E7%AB%A0-%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E6%A1%86%E6%9E%B6/05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)
- 跨书 RAG:[02-RAG原理深入](../../Happy-LLM/07-%E7%AC%AC7%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/02-RAG%E5%8E%9F%E7%90%86%E6%B7%B1%E5%85%A5.md)

---

## 📚 参考文献

[1] **MCP**: Model Context Protocol - https://modelcontextprotocol.io/

[2] **A2A**: Agent-to-Agent Protocol - Google https://github.com/google/A2A

[3] **ANP**: Agent Network Protocol - https://github.com/agent-network-protocol/

---

⬅ [04-ANP协议简介](04-ANP%E5%8D%8F%E8%AE%AE%E7%AE%80%E4%BB%8B.md)	|	➡ [进入第 11 章](../11-%E7%AC%AC11%E7%AB%A0-Agentic-RL/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
