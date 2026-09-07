---
tags: [Hello-Agents, 第9章, 长程Agent, 代码库维护, 总结]
chapter: 9
section: 9.6-9.7
---

# 9.6-9.7 长程 Agent 实战:代码库维护助手 + 章节小结

⬅ [04-TerminalTool文件访问](04-TerminalTool%E6%96%87%E4%BB%B6%E8%AE%BF%E9%97%AE.md)	|	➡ [进入第 10 章](../10-%E7%AC%AC10%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## 🎬 实战目标:**代码库维护助手**

> 一个能**自主维护中型代码库**的 Agent —— 综合运用本章所有工具。

### 任务示例

```
用户:"帮我审计这个项目,找出潜在 bug、补充测试、写优化建议"

Agent:
   1. 探索项目结构 (TerminalTool)
   2. 找出异常处理、TODO、FIXME (grep)
   3. 检索相关知识 (RAGTool)
   4. 记录发现 (NoteTool)
   5. 综合上下文 (ContextBuilder)
   6. 生成审计报告

→ 整个过程可能持续几小时, 跨越多次对话, 但 Agent 始终保持连贯
```

---

## 🏗 9.6.1 系统架构设计

```
                  代码库维护 Agent
                          │
       ┌──────────────────┼─────────────────┐
       │                  │                 │
   感知探索            理解推理           记录沉淀
       │                  │                 │
   TerminalTool        LLM 大脑          NoteTool
   (代码访问)         + ContextBuilder    (持久化笔记)
                      + MemoryTool
                      + RAGTool
```

### 6 大模块协同

| 模块 | 职责 |
|---|---|
| **TerminalTool** | 探索代码结构 / grep / read_file |
| **NoteTool** | 记录 task_state / conclusion / blocker / action |
| **MemoryTool** | 短期对话上下文 + 用户偏好 |
| **RAGTool** | 检索代码规范 / 最佳实践 |
| **ContextBuilder** | GSSC 流水线智能注入 |
| **LLM** | 综合推理 + 决策 |

---

## 🐍 9.6.2 核心实现

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import TerminalTool, NoteTool, MemoryTool, RAGTool
from hello_agents.context import ContextBuilder, ContextConfig


class CodebaseMaintainerAgent(SimpleAgent):
    """代码库维护 Agent —— 综合长程能力"""

    SYSTEM_PROMPT = """你是代码库维护助手。可用工具:

📁 TerminalTool: 探索代码(list_dir/read_file/grep/glob)
   ⚠️ 必须 **渐进式探索**: 先 list_dir, 再 head, 最后 grep 定位
   ⚠️ **绝不**一次性读所有文件

📝 NoteTool: 持久化记录
   - task_state: 当前进度
   - conclusion: 阶段结论
   - blocker:   遇到的问题
   - action:    下一步行动

🧠 MemoryTool: 短期记忆 + 用户偏好

📚 RAGTool: 检索代码规范 / 最佳实践

工作流程:
1. 接收任务 → 拆解
2. 用 NoteTool 记录初始计划 (task_state)
3. 用 TerminalTool 渐进探索
4. 关键发现 → NoteTool 沉淀
5. 综合分析 → 输出报告"""

    def __init__(self, name, llm, project_path):
        super().__init__(name=name, llm=llm, system_prompt=self.SYSTEM_PROMPT)

        # 初始化全部工具
        self.terminal = TerminalTool(workspace=project_path)
        self.notes    = NoteTool(workspace=f"{project_path}/.agent_notes")
        self.memory   = MemoryTool(user_id="developer")
        self.rag      = RAGTool(knowledge_base_path="./code_best_practices_kb")

        # 注册到 Registry
        registry = ToolRegistry()
        for tool in [self.terminal, self.notes, self.memory, self.rag]:
            registry.register_tool(tool)
        self.tool_registry = registry

        # ContextBuilder
        self.context_builder = ContextBuilder(
            memory_tool=self.memory,
            rag_tool=self.rag,
            config=ContextConfig(
                max_tokens=4000,
                reserve_ratio=0.2,
                relevance_weight=0.7,
                recency_weight=0.3,
            ),
        )

        self.history = []

    def run(self, user_input):
        # 1. 从笔记检索相关上下文
        relevant_notes = self.notes.run({
            "action": "search",
            "query": user_input,
            "limit": 3,
        })

        # 2. 转成 ContextPacket
        from hello_agents.context import ContextPacket
        from datetime import datetime
        note_packets = [
            ContextPacket(
                content=f"[Note: {n['title']}] {n['content']}",
                timestamp=datetime.fromisoformat(n['updated_at']),
                token_count=len(n['content']),
                relevance_score=0.75,
                metadata={"type": "note", "note_type": n['type']},
            )
            for n in relevant_notes
        ]

        # 3. ContextBuilder 智能构建
        optimized_context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.history,
            system_instructions=self.system_prompt,
            custom_packets=note_packets,
        )

        # 4. LLM + 工具循环
        # (实际会有多轮工具调用,这里简化)
        response = self.llm.invoke([
            {"role": "system", "content": optimized_context},
            {"role": "user", "content": user_input},
        ])

        # 5. 更新历史
        from hello_agents.core.message import Message
        self.history.extend([
            Message(user_input, "user"),
            Message(response, "assistant"),
        ])

        return response
```

---

## 🎮 9.6.3 真实运行示例(简化版)

### Day 1: 初次审计

```
用户: "帮我审计 ./my_project 这个项目"

Agent 内部流程:
  → list_dir(.)                    # 看根目录
  → read_file(README.md, head=50)  # 读项目说明
  → list_dir(src/)                  # 看源码
  → glob("**/*.py")                 # 找所有 Python 文件
  → grep("TODO|FIXME", "src/")      # 找待办

Agent 写笔记:
  📝 [task_state] 完成项目初步审计
  📝 [conclusion] 发现 23 个 TODO,主要集中在 src/utils/
  📝 [blocker]   错误处理不完整,15 处 raise 但无 try
  📝 [action]    1. 补充测试 2. 完善错误处理 3. 整理 TODO

回复用户: "完成初步审计,主要发现 3 个问题..."
```

### Day 3: 接着干

```
用户: "OAuth 那个问题怎样了?"

Agent 内部流程:
  → NoteTool.search("OAuth")        # 找之前笔记
  → 发现 Day 1 没记 OAuth
  → 答: "我之前的审计没专门记 OAuth,需要我现在查吗?"

用户: "查一下"

Agent:
  → grep("OAuth", "src/")
  → read_file(src/auth/oauth.py)
  → 分析问题
  → NoteTool.create([blocker] "OAuth token 刷新逻辑有竞态条件")
  → 答: "找到了, OAuth token 刷新有竞态..."
```

→ **跨 3 天**,**Agent 状态持续连贯**。

---

## 📊 9.6.4 运行效果分析

### 上下文工程的价值体现

| 指标 | 无上下文工程 | **有上下文工程** |
|---|---|---|
| 单轮 token 消耗 | 50K+(预读全部代码) | **3K**(精挑细选) |
| 跨会话连贯性 | 0%(完全忘了) | **90%+**(NoteTool 持久化) |
| 探索效率 | 低(盲读) | **高**(JIT 按需) |
| Bug 召回率 | 60% | **85%**(系统化搜索) |
| 成本 | 高 | **低 10 倍** |

---

## 📝 9.7 章节小结

### 核心收获

#### 收获 1:**理论层面**(认知突破)
- **上下文不是越多越好**(Context Rot 现象)
- **LLM 有"注意力预算"**(有限资源 + 边际收益递减)
- **JIT 上下文 > 预加载**

#### 收获 2:**工程实践**
- **ContextBuilder** 的 GSSC 流水线
- **NoteTool** 长程外脑
- **TerminalTool** 即时文件访问

#### 收获 3:**长时程任务 3 大手段**
- **压缩整合**:对话接力
- **结构化笔记**:迭代式开发(NoteTool)
- **子代理架构**:复杂分析(第 11 章会回归)

#### 收获 4:**5 大工具协同**
```
TerminalTool + NoteTool + MemoryTool + RAGTool + ContextBuilder
   ↓
HelloAgents 完整长程能力
```

---

## 🎯 9.7.1 设计哲学回顾

### 哲学 1:**有限即设计**
- 不假设无限上下文
- 工程化处理"有限注意力预算"

### 哲学 2:**结构化即可解释**
- 5 分区模板(`[Role]`/`[Task]`/`[Evidence]`/`[Context]`/`[Output]`)
- 笔记的 YAML+Markdown 格式
- → **可调试、可 A/B 测试、可演进**

### 哲学 3:**JIT 即智能**
- 不预加载,**按需检索**
- 类似人类:**目录 → 文件名 → 时间戳 → 真正读**

### 哲学 4:**外脑即长程**
- NoteTool 持久状态
- TerminalTool 文件作为状态
- → 上下文窗口只装"当前必要"

---

## 🚀 9.7.2 接下来怎么走?

```
本章 (上下文工程)
   ↓
第 10 章 (通信协议 MCP/A2A/ANP)
   ↓ 让 Agent 互联互通
第 11 章 (Agentic-RL)
   ↓ 让 Agent 自主进化
第 12 章 (Agent 评估)
   ↓ 衡量 Agent 能力
```

---

## ⚠️ 整章踩坑清单

| 坑 | 解法 |
|---|---|
| 觉得"上下文越长越好" | 看 9.2 上下文腐蚀 |
| ContextBuilder `min_relevance` 设太高 | 一般 0.1-0.3 |
| NoteTool 所有事都记 `general` | 用对类型 |
| TerminalTool 不加 `head/tail` | 大文件爆 token |
| 没有 NoteTool 想做长程任务 | 必须用 |
| 子代理盲目嵌套 | 关注点分离才有用 |

---

## ❓ 本章自测题

→ 见 [自测题汇总-第 9 章](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#%E7%AC%AC-9-%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B)

---

## 📌 本章一句话总结

**上下文工程让 Agent 从"短跑"升级为"马拉松"** —— 通过 **ContextBuilder + NoteTool + TerminalTool**,HelloAgents 真正能做几天的长程任务。

---

## 🚀 下一章预告

**第 10 章 智能体通信协议** —— 让 Agent **互联互通**:
- **MCP**(Model Context Protocol):**模型 ↔ 工具/数据**(2024 大热)
- **A2A**(Agent-to-Agent):**Agent ↔ Agent**
- **ANP**(Agent Network Protocol):**Agent ↔ 网络发现**

→ 这是**当下 Agent 圈最前沿的话题**。

---

## 🔗 延伸阅读

- 上一节:[04-TerminalTool文件访问](04-TerminalTool%E6%96%87%E4%BB%B6%E8%AE%BF%E9%97%AE.md)
- **下一章**:[00-章节总览](../10-%E7%AC%AC10%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 子代理架构(第 11 章会展开):多智能体强化学习

---

⬅ [04-TerminalTool文件访问](04-TerminalTool%E6%96%87%E4%BB%B6%E8%AE%BF%E9%97%AE.md)	|	➡ [进入第 10 章](../10-%E7%AC%AC10%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
