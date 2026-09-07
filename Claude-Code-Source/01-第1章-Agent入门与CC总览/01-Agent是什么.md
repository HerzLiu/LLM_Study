---
tags: [Claude-Code, 第1章, Agent入门, 核心概念]
chapter: 1
section: 1.1
---

# 1.1 Agent 是什么？和 Chatbot 的本质区别

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-Claude-Code是什么](02-Claude-Code%E6%98%AF%E4%BB%80%E4%B9%88.md)

> ⭐ 这是整个系列的"地基概念"，理解了它，后面 7 章都是顺水推舟。

---

## 🎬 故事比喻：从"自动售货机"到"实习生"

想象你想喝一杯冰美式。

| 角色 | 行为 | 类比 |
|---|---|---|
| **自动售货机** | 你按"冰美式"按钮，它吐杯冰美式。按啥出啥，**没有判断** | 普通程序 |
| **ChatGPT 网页** | 你说"我想提神"，它说"建议你喝冰美式"。会说话，**不会真的冲咖啡** | Chatbot |
| **会做事的实习生** | 你说"我想提神"，他**自己**判断 → 拿杯 → 接咖啡 → 加冰 → 端给你 | Agent |

**实习生这一连串行为里发生了什么？**

1. **思考**：今天热，提神 → 应该是冰美式
2. **行动**：走到咖啡机 → 拿杯子 → 接咖啡 → 加冰
3. **观察**：检查咖啡是否冒泡、是否溢出
4. **再思考**：杯子还差一点满，再加一点
5. **交付**：把咖啡端给你

**Agent ≈ 这个会"思考-行动-观察-再思考"的实习生。**
区别在于：实习生用真手脚、真大脑；Agent 用 **工具**（Tool）和 **大语言模型**（LLM）。

---

## 🔧 核心概念：一句话定义

> **Agent = LLM + Tools + Loop**

| 成分 | 角色 | 比喻 |
|---|---|---|
| **LLM**（大语言模型） | 决策大脑 | 实习生的脑子 |
| **Tools**（工具集） | 与世界交互的能力 | 手、脚、眼、嘴 |
| **Loop**（循环） | 思考→行动→观察→再思考 | 实习生一边干一边想 |

**关键差别**：
- **Chatbot** 只有 LLM，输出文字
- **Agent** 在 LLM 外面套一层"循环 + 工具调用"，让模型能**动手做事**

---

## 📊 对比表：4 种"会聊天的程序"

| 形态 | 形态 | 会动手吗？ | 例子 |
|---|---|---|---|
| **规则机器人** | 关键词触发 | 否 | 客服 FAQ 机器人 |
| **Chatbot** | 纯 LLM 对话 | 否 | ChatGPT 网页版 |
| **半 Agent** | LLM + 单一工具 | 半（仅生成代码） | GitHub Copilot |
| **完整 Agent** | LLM + 工具集 + 循环 | 是 | Claude Code / Cursor |

---

## ⭐ 加深理解：为什么"循环"是关键？

很多人以为"Agent = 给 LLM 加工具"。**错。**

加工具但**不加循环**，LLM 只能调一次工具就结束。但真实任务往往要**多次工具调用**：

```
用户："修一下 utils.py 的 bug"

Agent 实际要做的：
  第 1 轮：Read utils.py → 读到代码
  第 2 轮：Read tests/test_utils.py → 看测试预期
  第 3 轮：Edit utils.py → 改代码
  第 4 轮：Bash pytest → 跑测试
  第 5 轮：发现还有错 → Edit 再改
  第 6 轮：Bash pytest → 通过
  第 7 轮：告诉用户"修好了"
```

**循环让 Agent 能"基于上一轮结果决定下一轮做啥"**。这才是"自主"的本质。

---

## 🐍 30 秒看代码骨架

```python
def run_agent(user_input):
    messages = [{"role": "user", "content": user_input}]
    while True:                              # ⭐ 关键：循环
        response = call_llm(messages)        # LLM 决策
        if response.is_done:                 # 它说"完事了"
            break
        if response.has_tool_call:           # 它要调工具
            result = run_tool(response.tool) # 真去执行
            messages.append(result)          # 结果塞回历史
        # 进入下一轮
```

完整可运行版本：[04-30行Python假Agent](04-30%E8%A1%8CPython%E5%81%87Agent.md)

---

## ⚠️ 小白避坑

1. **"Agent" 不是某种神秘技术**
   - 它是一种**架构**：LLM + 循环 + 工具
   - 任何 LLM（GPT、Claude、本地模型）都能做 Agent
2. **不要把"框架"和"Agent"画等号**
   - LangChain、AutoGen 是 Agent 框架
   - 但你**完全可以不用框架手撸 Agent**（本系列第 8 章就是手撸版）
3. **Agent 也会"翻车"**
   - 死循环、瞎调工具、撒谎报告完成
   - 所以需要"防御性设计"（轮次上限、权限、验证），后面章节细讲

---

## 🔗 延伸阅读

- 下一节：[02-Claude-Code是什么](02-Claude-Code%E6%98%AF%E4%BB%80%E4%B9%88.md) —— 一个具体的 Agent 长啥样
- 跳到主循环：[00-章节总览](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

---

## ❓ 小测验

> 在心里复述（不要看回去）：
> 1. Agent 和 Chatbot 的本质区别是什么？（提示：3 个字）
> 2. Agent 三大成分是什么？
> 3. "循环"在 Agent 中扮演什么角色？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#11-agent-%E6%98%AF%E4%BB%80%E4%B9%88)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-Claude-Code是什么](02-Claude-Code%E6%98%AF%E4%BB%80%E4%B9%88.md)
