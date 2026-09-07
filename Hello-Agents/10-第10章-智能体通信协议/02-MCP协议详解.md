---
tags: [Hello-Agents, 第10章, MCP, Anthropic, 重点]
chapter: 10
section: 10.2
---

# 10.2 MCP 协议详解

⬅ [01-为什么需要通信协议](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE.md)	|	➡ [03-A2A协议实战](03-A2A%E5%8D%8F%E8%AE%AE%E5%AE%9E%E6%88%98.md)

> ⭐ **本章重点**:MCP 是当下**最热**的 Agent 协议,**生态最成熟**,**必懂**。

---

## 🎬 故事比喻:智能体的"USB-C"

```
你的 Agent 要做这些:
   - 读本地文件
   - 查 PostgreSQL
   - 搜 GitHub 代码
   - 发 Slack 消息
   - 访问 Google Drive

❌ 传统:每个服务写一个适配器(累 + 不兼容)
✅ MCP:统一接入方式(像 USB-C 充电)
```

→ **MCP 之于 Agent**,**正如 USB-C 之于充电**。

---

## 🏗 10.2.1 MCP 三层架构

> MCP 采用 **Host / Client / Server** 三层架构。

### 场景演示:Claude Desktop 问"我桌面有哪些文档?"

```
┌───────────────────────────────────────────┐
│  1. Host (宿主)                            │
│     Claude Desktop                          │
│     - 接收用户提问                           │
│     - 与 Claude 模型交互                     │
└───────────────────┬───────────────────────┘
                    │ Claude 决定调用工具
                    ▼
┌───────────────────────────────────────────┐
│  2. Client (客户端)                         │
│     Host 内置的 MCP Client                  │
│     - 与适当的 MCP Server 建立连接           │
│     - 发送请求,接收响应                       │
└───────────────────┬───────────────────────┘
                    │ MCP 协议通信
                    ▼
┌───────────────────────────────────────────┐
│  3. Server (服务器)                         │
│     文件系统 MCP Server                     │
│     - 执行实际文件扫描                       │
│     - 返回桌面文档列表                       │
└───────────────────────────────────────────┘
```

### 完整交互流程

```
用户问题 → Claude Desktop(Host) → Claude 模型分析
   → 需要文件信息
   → MCP Client 连接
   → 文件系统 MCP Server
   → 执行操作
   → 返回结果
   → Claude 生成回答
   → 显示给用户
```

### 三层架构的优势:**关注点分离**

| 层 | 关注 |
|---|---|
| **Host** | 用户体验 |
| **Client** | 协议通信 |
| **Server** | 具体功能实现 |

→ **开发者只需写 MCP Server**,Host 和 Client **不用管**。

---

## 🎯 10.2.2 MCP 三大核心能力

| 能力 | 性质 | 用途 | 举例 |
|---|---|---|---|
| **Tools(工具)** | **主动** | 执行操作 | 调用 API、写文件、发邮件 |
| **Resources(资源)** | **被动** | 提供数据 | 读取文件内容、查询数据库 |
| **Prompts(提示)** | **指导性** | 提供模板 | 代码审查模板、报告模板 |

→ 三种能力**互补**:Tools 做事 / Resources 提供数据 / Prompts 指导流程。

---

## 🔄 10.2.3 MCP 工作流程

### 关键问题:**LLM 怎么决定用哪个工具?**

```
1. 工具发现
   MCP Client 调 list_tools() → 获取所有可用工具描述

2. 上下文构建
   工具列表加到系统提示词:
   "你可以使用以下工具:
    - read_file(path: str): 读取文件
    - search_code(query, language): 搜索代码
    ..."

3. 模型推理
   LLM 分析用户问题 + 工具描述 → 决定调用

4. 工具执行
   通过 MCP Server 执行 → 获取结果

5. 结果整合
   LLM 基于工具结果生成最终回答
```

→ **完全自动化** —— LLM 根据**工具描述质量**决定调用。
→ **工具 description 至关重要**(详见 [[../07-第7章-构建你的智能体框架/05-工具系统#🔑 1. **`name` + `description`**|7.5 工具系统]])。

---

## 🆚 10.2.4 MCP vs Function Calling

> 很多人混淆这两者,**它们其实是互补关系**。

### 类比

| | Function Calling | **MCP** |
|---|---|---|
| 比喻 | "**你学会了打电话**" | "**全球电话标准**" |
| 角色 | **LLM 的能力**(决定何时调用、生成参数) | **基础设施协议**(工具如何描述、如何调用) |
| 关系 | 互补,**Function Calling 用 MCP 作为通信层** | |

### 代码对比

#### Function Calling(传统方式)

```python
# 为每个 LLM 提供商定义不同格式
# OpenAI 格式
openai_tools = [{
    "type": "function",
    "function": {
        "name": "search_github",
        "description": "搜索 GitHub 仓库",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]

# Claude 格式(略不同)
claude_tools = [{
    "name": "search_github",
    "input_schema": {  # 不是 parameters
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}]

# 自己实现工具函数
def search_github(query):
    response = requests.get("https://api.github.com/search/repositories",
                            params={"q": query})
    return response.json()

# 处理不同模型的响应格式(各种 if-else)
```

#### MCP(标准方式)

```python
from hello_agents.protocols import MCPClient

# 连接社区提供的 MCP 服务器(无需自己实现!)
github_client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-github"])

async with github_client:
    # 自动发现工具
    tools = await github_client.list_tools()

    # 调用工具(标准化接口)
    result = await github_client.call_tool(
        "search_repositories",
        {"query": "AI agents"},
    )

# 任何支持 MCP 的模型(OpenAI/Claude/Llama)都用同一套代码
```

→ **MCP 的胜利在于"标准化 + 复用"**。

---

## 🛠 10.2.5 使用 MCP 客户端(异步 API)

### 连接 MCP 服务器

```python
import asyncio
from hello_agents.protocols import MCPClient

async def connect_to_server():
    # 方式 1: 连社区文件系统服务器
    # npx 自动下载并运行
    client = MCPClient([
        "npx", "-y",
        "@modelcontextprotocol/server-filesystem",
        ".",
    ])

    async with client:   # ⭐ 用 async with 确保正确关闭
        tools = await client.list_tools()
        print(f"可用工具: {[t['name'] for t in tools]}")

    # 方式 2: 连自定义 Python 服务器
    client = MCPClient(["python", "my_mcp_server.py"])
    async with client:
        pass

asyncio.run(connect_to_server())
```

### 发现工具

```python
async def discover_tools():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        tools = await client.list_tools()
        print(f"服务器提供 {len(tools)} 个工具:")

        for tool in tools:
            print(f"\n名称: {tool['name']}")
            print(f"描述: {tool.get('description', '无描述')}")

            # 打印参数
            if 'inputSchema' in tool:
                for name, info in tool['inputSchema'].get('properties', {}).items():
                    print(f"  - {name} ({info.get('type')}): {info.get('description', '')}")
```

输出示例:
```
服务器提供 5 个工具:

名称: read_file
描述: 读取文件内容
  - path (string): 文件路径

名称: write_file
描述: 写入文件内容
  - path (string): 文件路径
  - content (string): 文件内容
```

### 调用工具

```python
async def use_tools():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        # 读文件
        result = await client.call_tool("read_file", {"path": "README.md"})

        # 列目录
        result = await client.call_tool("list_directory", {"path": "."})

        # 写文件
        result = await client.call_tool("write_file", {
            "path": "output.txt",
            "content": "Hello from MCP!",
        })
```

---

## 📡 10.2.6 MCP 5 种传输方式

> **MCP 是传输层无关**(Transport Agnostic),**协议本身**不依赖特定通信通道。

### 5 种方式对比

| 传输方式 | 适用场景 | 特点 |
|---|---|---|
| **Memory** | 单元测试 / 快速原型 | 内存中,**最简** |
| **Stdio** | 本地开发 / 调试 / Python 脚本 | **最常用** |
| **HTTP** | 生产环境 / 远程服务 / 微服务 | 标准 HTTP |
| **SSE** | 实时通信 / 流式处理 / 长连接 | Server-Sent Events |
| **StreamableHTTP** | 双向流式 HTTP 场景 | 高级 |

### 使用方式快速对比

```python
from hello_agents.tools import MCPTool
from hello_agents.protocols import MCPClient

# 1. Memory:不指定参数,使用内置演示服务器
mcp_tool = MCPTool()

# 2. Stdio:本地 Python 服务器
mcp_tool = MCPTool(server_command=["python", "my_server.py"])

# 3. Stdio + npx:社区服务器
mcp_tool = MCPTool(server_command=[
    "npx", "-y", "@modelcontextprotocol/server-filesystem", ".",
])

# 4. HTTP:远程服务器
async with MCPClient("http://api.example.com/mcp") as client:
    tools = await client.list_tools()

# 5. SSE:流式
async with MCPClient("http://localhost:8080/sse", transport_type="sse") as client:
    result = await client.call_tool("stream_process", {...})
```

→ **入门用 Memory / Stdio,生产用 HTTP / SSE**。

---

## 🤖 10.2.7 在 Agent 中使用 MCP(MCPTool)

### MCP 工具的"自动展开"机制 ⭐

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

# 添加 MCP 工具(自动展开!)
mcp_tool = MCPTool(name="calculator")
agent.add_tool(mcp_tool)
# ✅ MCP 工具 'calculator' 已展开为 6 个独立工具
# - calculator_add
# - calculator_subtract
# - calculator_multiply
# - calculator_divide
# - calculator_greet
# - calculator_get_system_info

# Agent 像调用普通工具一样调用
response = agent.run("计算 25 乘以 16")
print(response)   # 25 × 16 = 400
```

### 多 MCP 服务器:**用 `name` 避免冲突**

```python
agent = SimpleAgent(name="文件助手", llm=HelloAgentsLLM())

# 文件系统服务器
fs_tool = MCPTool(
    name="fs",   # ⭐ 唯一前缀
    server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."],
)
agent.add_tool(fs_tool)
# 展开为 fs_read_file, fs_write_file, ...

# 自定义服务器
custom_tool = MCPTool(
    name="custom",
    server_command=["python", "my_mcp_server.py"],
)
agent.add_tool(custom_tool)

# Agent 自动选择正确工具
response = agent.run("读取 README.md 并总结")
```

---

## 🌍 10.2.8 MCP 社区生态

> MCP 的杀手锏:**大量现成的 MCP 服务器**!

### 三大资源库

| 资源 | 链接 |
|---|---|
| **Awesome MCP Servers** | https://github.com/punkpeye/awesome-mcp-servers |
| **MCP Servers 网站** | https://mcpservers.org/ |
| **官方 MCP Servers** | https://github.com/modelcontextprotocol/servers |

### 常用官方 MCP 服务器

| 服务器 | 用途 |
|---|---|
| **filesystem** | 文件系统访问 |
| **github** | GitHub 仓库/Issue/PR |
| **postgres** | PostgreSQL 数据库 |
| **slack** | Slack 消息 |
| **google-drive** | Google Drive |
| **memory** | 知识图谱记忆 |
| **fetch** | 网页抓取 |
| **time** | 时间和时区 |

### 有趣的 Use Case

#### 用例 1:**自动化网页测试**(Playwright MCP)

```python
playwright_tool = MCPTool(
    name="playwright",
    server_command=["npx", "-y", "@playwright/mcp"],
)
# Agent 可以:
# - 打开浏览器访问网站
# - 填表单
# - 截图验证
# - 生成测试报告
```

#### 用例 2:**智能笔记助手**(Obsidian + Perplexity)

```python
# Agent 可以:
# - 搜索最新技术资讯
# - 整理结构化笔记
# - 保存到 Obsidian
# - 自动建笔记间链接
```

#### 用例 3:**项目管理自动化**(Jira + GitHub)

```python
# Agent 可以:
# - 从 GitHub Issue 创建 Jira 任务
# - 同步代码提交到 Jira
# - 自动更新 Sprint 进度
```

---

## 🎮 10.2.9 实战:多 Agent 协作文档助手

```python
"""
多 Agent 协作的智能文档助手
- Agent1: GitHub 搜索专家
- Agent2: 文档生成专家
"""
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

# === Agent 1: GitHub 搜索专家 ===
github_searcher = SimpleAgent(
    name="GitHub 搜索专家",
    llm=HelloAgentsLLM(),
    system_prompt="搜索 GitHub 仓库,返回结构化结果",
)
github_searcher.add_tool(MCPTool(
    name="gh",
    server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
))

# === Agent 2: 文档生成专家 ===
document_writer = SimpleAgent(
    name="文档生成专家",
    llm=HelloAgentsLLM(),
    system_prompt="根据信息生成 Markdown 报告",
)
document_writer.add_tool(MCPTool(
    name="fs",
    server_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."],
))

# === 执行任务 ===
# 1. 搜索
search_results = github_searcher.run("搜索关于 'AI agent' 的 GitHub 仓库,前 5 个最相关")

# 2. 生成报告
report_content = document_writer.run(f"""
根据以下 GitHub 搜索结果,生成 Markdown 研究报告:
{search_results}
""")

# 3. 保存
with open("report.md", "w") as f:
    f.write(report_content)
print("✅ 报告已保存")
```

→ **多 Agent + MCP** 协作,**几十行代码搞定完整工作流**。

---

## ⚠️ 小白避坑

1. **`async with` 必须**
   - 不用 → 连接泄漏
2. **多 MCP 服务器必加 `name` 前缀**
   - 防工具名冲突
3. **GitHub MCP 需要 token**
   - 环境变量 `GITHUB_PERSONAL_ACCESS_TOKEN`
4. **`npx` 第一次慢**
   - 需要下载包,**之后会快**
5. **工具描述质量决定一切**
   - 写自定义 MCP Server 时,**description 写好**

---

## 📌 10.2 节要点

| 知识点 | 一句话 |
|---|---|
| **MCP 定位** | 智能体的 USB-C —— 统一接入工具 |
| **三层架构** | Host(用户)+ Client(协议)+ Server(功能) |
| **3 大能力** | Tools(操作)+ Resources(数据)+ Prompts(模板) |
| **vs Function Calling** | **互补**:FC 是能力,MCP 是协议 |
| **MCPTool 自动展开** | 一个 MCPTool → 多个独立工具 |
| **5 种传输** | Memory / Stdio / HTTP / SSE / StreamableHTTP |
| **生态** | **数百个社区 MCP 服务器**,即装即用 |

---

## 🔗 延伸阅读

- 上一节:[01-为什么需要通信协议](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE.md)
- 下一节:[03-A2A协议实战](03-A2A%E5%8D%8F%E8%AE%AE%E5%AE%9E%E6%88%98.md)
- HelloAgents 工具:[05-工具系统](../07-%E7%AC%AC7%E7%AB%A0-%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E6%A1%86%E6%9E%B6/05-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F.md)

---

⬅ [01-为什么需要通信协议](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE.md)	|	➡ [03-A2A协议实战](03-A2A%E5%8D%8F%E8%AE%AE%E5%AE%9E%E6%88%98.md)
