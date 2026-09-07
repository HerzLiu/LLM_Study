---
tags: [Claude-Code, 第6章, 上下文隔离, 动机]
chapter: 6
section: 6.1
---

# 6.1 为什么需要 SubAgent（上下文隔离）

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-AgentTool与内置Agent](02-AgentTool%E4%B8%8E%E5%86%85%E7%BD%AEAgent.md)

---

## 🎬 故事比喻：主管的工作台

你是主管 A，老板让你做一份"竞品调研报告"，涉及：
- 查 5 个竞品官网
- 读 5 份财报 PDF
- 整理 SWOT
- 写报告

### 方案 1：你一个人干

```
工作台堆满：
- 5 个网站打开的浏览器（30 个 tab）
- 5 份 PDF 平铺
- 一堆便签写关键信息
- 自己思考的草稿
- ……

写报告时：脑子里全是 PDF 截图，写不出重点
```

### 方案 2：派 5 个实习生

```
你的工作台：
- 5 张便签，每张写一句"竞品 A 的护城河是数据"
- 一份空白报告等填

每个实习生的工作台：
- 5 个浏览器 tab + 1 份 PDF
- 自己整理一上午
- 最后给主管一句话总结
```

**主管脑子清爽 → 报告质量高**。

---

## 🔧 SubAgent 的核心动机

| 维度 | 自己干 | 派 SubAgent |
|---|---|---|
| 主 Agent context | 爆炸 | 清爽 |
| 工具调用细节 | 全在主 context | 在 SubAgent context |
| 中间数据 | 占 token | 不占主 token |
| 综合判断 | 信息淹没 | 基于摘要 |

**核心**：**上下文隔离**。

---

## 🔢 量化对比

假设任务：找出代码库里所有认证相关文件。

### 方案 A：主 Agent 自己 grep

```
grep "auth" → 200 行结果（~3000 tokens）进 messages
grep "login" → 150 行 (~2500 tokens) 进 messages
grep "session" → 180 行 (~3000 tokens) 进 messages
read auth.py → 500 行 (~5000 tokens) 进 messages
read login.py → 400 行 (~4000 tokens) 进 messages
...
最终消耗 ~50K-100K tokens 在主 context
```

### 方案 B：派 Explore SubAgent

```
主 Agent 发：
  Task(subagent_type=Explore,
       prompt="Find auth-related files, return summary under 200 words")
  ↓
Explore SubAgent 在它自己的 context 里 grep + read（消耗 50K tokens）
  ↓
SubAgent 返回："Found 5 files: auth.py, login.py, ..."（200 tokens）
  ↓
主 Agent context 只多了 ~300 tokens
```

**省下 99% 主 context！**

---

## ⭐ 加深理解：不只是省 token，还有专门化

每个 SubAgent 可以有：
- **自己的 system prompt**（更窄、更专）
- **自己的工具集**（最小化权限）
- **自己的模型选择**（便宜的够用就别用贵的）

类比：
- **Explore SubAgent** = 实习搜索员，只给 Read/Grep/Glob
- **Verify SubAgent** = 实习测试员，只给 Read + Bash(pytest only)
- **Code Reviewer SubAgent** = 实习评审员，只读不写

**专门化 + 隔离 = 1+1 > 2**。

---

## 🐍 直觉对比代码

```python
# ============= 方案 A: 主 Agent 自己干 =============
async def main_agent_alone():
    messages = []
    # 主 Agent 自己跑搜索
    r1 = await grep("auth")            # 3000 tokens 进 messages
    messages.append(make_tool_result(r1))
    r2 = await read("auth.py")         # 5000 tokens 进 messages
    messages.append(make_tool_result(r2))
    # ... 总共 50K tokens
    response = await call_llm(messages) # 现在 messages 已经很长

# ============= 方案 B: 派 SubAgent =============
async def main_agent_with_sub():
    messages = []
    # 主 Agent 派 SubAgent
    summary = await spawn_subagent(
        type="Explore",
        prompt="Find auth files. Report in 200 words."
    )
    # summary 是字符串，只有 ~300 tokens
    messages.append(make_tool_result(summary))
    response = await call_llm(messages) # messages 仍然短

# spawn_subagent 内部:
async def spawn_subagent(type, prompt):
    sub_messages = []  # ⭐ 独立的 context！
    # SubAgent 自己干所有事（grep / read / 分析）
    # 消耗的 token 都在 sub_messages 里
    # 最后返回最终字符串
    return final_summary  # 只回一个字符串
```

---

## ⚠️ 小白避坑

1. **SubAgent 不是 "Agent 框架"**
   - 它就是"嵌套的同一个 query() 函数"
   - 用同一份代码递归启动

2. **SubAgent 的开销**
   - 启动有成本（prompt cache 可能 miss）
   - 任务太小（< 5 个工具调用）不建议派

3. **"分工" ≠ "上下文隔离"**
   - 你可以让一个主 Agent 又写代码又测试（分工是逻辑层面）
   - SubAgent 解决的是**物理 context 隔离**

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-AgentTool与内置Agent](02-AgentTool%E4%B8%8E%E5%86%85%E7%BD%AEAgent.md) —— 具体实现
- 上下文管理：[第 4 章](../04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 真源码：`_source/.../src/tools/AgentTool/`

---

## ❓ 小测验

> 1. SubAgent 的核心动机是什么？（提示：4 个字）
> 2. 派 SubAgent 后，主 Agent 的 context 增加多少？
> 3. SubAgent 除了省 token 还有什么好处？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#61-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81-subagent)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-AgentTool与内置Agent](02-AgentTool%E4%B8%8E%E5%86%85%E7%BD%AEAgent.md)
