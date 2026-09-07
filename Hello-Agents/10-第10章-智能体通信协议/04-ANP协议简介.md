---
tags: [Hello-Agents, 第10章, ANP, 服务发现, 去中心化]
chapter: 10
section: 10.4
---

# 10.4 ANP 协议简介

⬅ [03-A2A协议实战](03-A2A%E5%8D%8F%E8%AE%AE%E5%AE%9E%E6%88%98.md)	|	➡ [05-自定义MCP服务器与小结](05-%E8%87%AA%E5%AE%9A%E4%B9%89MCP%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8E%E5%B0%8F%E7%BB%93.md)

> **ANP**(Agent Network Protocol)—— **大规模 Agent 网络**的服务发现协议。
> ⚠️ **概念性框架**,**生态早期**,主要了解思想。

---

## 🎬 故事比喻:Agent 圈的 DNS + 黄页

```
你想找一家"24 小时开门的麻辣火锅店"
   ↓
- 用百度地图(服务发现)
- 看评分 + 距离(路由)
- 拨打电话(连接)
```

→ ANP 之于 Agent 网络,**就像 DNS + 黄页之于现实世界**。

---

## 🎯 10.4.1 ANP 解决的问题

### 场景:**几百个 Agent 的网络**

```
Agent 1: 翻译专家
Agent 2: 数据分析专家
Agent 3: 代码生成专家
Agent 4: ...
Agent N: ...

你的 Agent: "我需要把这段中文翻译成英文,谁能干?"
   ↓
- MCP 答不了(它管工具,不管"找 Agent")
- A2A 答不了(它管对话,不管"先找谁")
- ANP 才能答: 服务发现
```

### ANP 的核心:**去中心化服务发现**

```
1. 服务注册
   每个 Agent 把自己的能力注册到网络

2. 服务发现
   其他 Agent 通过关键词/类型/标签查询

3. 路由
   找到匹配的 Agent → 建立连接
```

→ **不需要预先配置所有 Agent 连接关系**。

---

## 🏗 10.4.2 三协议层次关系

```
┌────────────────────────────────────────┐
│  ANP (Agent Network Protocol)          │
│  服务发现 + 网络拓扑                       │
│  "找到合适的 Agent"                       │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  A2A (Agent-to-Agent Protocol)         │
│  Agent ↔ Agent 通信                      │
│  "和找到的 Agent 对话"                    │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  MCP (Model Context Protocol)          │
│  Agent ↔ 工具/资源                       │
│  "Agent 调用具体能力"                     │
└────────────────────────────────────────┘
```

→ **三层完整协议栈**:**ANP 找 → A2A 谈 → MCP 用工具**。

---

## 🛠 10.4.3 ANP 基础使用

> HelloAgents 提供**概念性实现**,主要了解思路。

### 注册服务

```python
from hello_agents.tools import ANPTool

anp_tool = ANPTool()

# 注册计算器服务
anp_tool.run({
    "action": "register_service",
    "service_id": "calculator-001",
    "service_type": "math",
    "endpoint": "http://localhost:8080",
    "capabilities": ["addition", "subtraction", "multiplication"],
    "tags": ["fast", "reliable"],
})
```

### 发现服务

```python
# 发现所有服务
services = anp_tool.run({"action": "discover_services"})

# 按类型过滤
math_services = anp_tool.run({
    "action": "discover_services",
    "service_type": "math",
})

# 按能力过滤
multiply_services = anp_tool.run({
    "action": "discover_services",
    "capability": "multiplication",
})
```

### 服务路由

```python
# 找到能算乘法的 Agent → 调它
service = anp_tool.run({
    "action": "find_best_service",
    "capability": "multiplication",
    "criteria": "lowest_latency",  # 选延迟最低的
})

# 拿到 service.endpoint 后用 A2A 调
```

---

## ⚖️ 10.4.4 ANP 的实际状况

> ⚠️ **必须务实**:**生态最不成熟**

| 维度 | 状况 |
|---|---|
| 协议成熟度 | **早期概念** |
| 官方实现 | 部分(在演进) |
| 生产应用 | **极少** |
| 推荐用法 | **了解思想 + 等生态成熟** |

→ HelloAgents 的 ANPTool **是概念模拟**,不是生产级实现。

---

## 🌟 10.4.5 ANP 的未来场景设想

### 设想 1:**Agent App Store**

```
未来:数千个 Agent 服务在网络
   ↓ ANP
你的 Agent 自动发现并调用合适的
   ↓
Agent 经济崛起
```

### 设想 2:**企业内 Agent 网络**

```
公司内部:
   - HR Agent
   - 财务 Agent
   - 法务 Agent
   - 技术 Agent
   ↓ ANP 注册
新入职员工的 Agent → 自动发现并使用
```

### 设想 3:**跨组织 Agent 协作**

```
甲公司 Agent → 乙公司 Agent → 丙公司 Agent
   ↓ ANP 跨域发现
形成产业链级 Agent 网络
```

→ **真正的"Agent 互联网"**。

---

## 🔮 10.4.6 当下的替代方案

> ANP 还不成熟,**当下**可以用什么实现"服务发现"?

| 替代方案 | 说明 |
|---|---|
| **MCP 工具市场** | mcpservers.org 就是事实上的 MCP "应用商店" |
| **LangChain Hub** | 共享 prompt 和 chain |
| **自建注册中心** | 用 Consul / etcd / Redis |
| **关注 ANP 演进** | 协议成熟再用 |

---

## ⚠️ 小白避坑

1. **不要把 ANP 当生产工具**
   - 当下学概念即可
2. **如果有"找 Agent" 需求**
   - 用 MCP 工具市场 / 自建注册中心
3. **关注 Agent 圈的协议演进**
   - 2025-2026 年这部分会快速变化
4. **理解协议栈思想 > 死记 API**
   - ANP/A2A/MCP **三层协议栈**的关系是核心收获

---

## 📌 10.4 节要点

| 知识点 | 一句话 |
|---|---|
| **ANP 定位** | 大规模 Agent 网络的**服务发现协议** |
| **核心功能** | 服务注册 + 发现 + 路由 |
| **三协议栈** | ANP(找)+ A2A(谈)+ MCP(用) |
| **现状** | **概念性框架,生态最早期** |
| **当下替代** | MCP 工具市场 / 自建注册 |
| **未来想象** | **Agent App Store** / **Agent 经济** |

---

## 🔗 延伸阅读

- 上一节:[03-A2A协议实战](03-A2A%E5%8D%8F%E8%AE%AE%E5%AE%9E%E6%88%98.md)
- 下一节:[05-自定义MCP服务器与小结](05-%E8%87%AA%E5%AE%9A%E4%B9%89MCP%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8E%E5%B0%8F%E7%BB%93.md)
- 三协议对比:[10.1.4](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE.md#-1014-%E4%B8%89%E5%8D%8F%E8%AE%AE%E5%AE%8C%E6%95%B4%E5%AF%B9%E6%AF%94)

---

⬅ [03-A2A协议实战](03-A2A%E5%8D%8F%E8%AE%AE%E5%AE%9E%E6%88%98.md)	|	➡ [05-自定义MCP服务器与小结](05-%E8%87%AA%E5%AE%9A%E4%B9%89MCP%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8E%E5%B0%8F%E7%BB%93.md)
