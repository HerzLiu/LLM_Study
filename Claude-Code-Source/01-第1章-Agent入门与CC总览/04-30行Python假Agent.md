---
tags: [Claude-Code, 第1章, 代码实战, Python入门]
chapter: 1
section: 1.4
---

# 1.4 30 行 Python 跑一个"假" Agent

⬅ [03-最简核心循环](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md)　|	➡ [第 2 章 →](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)

> 🎯 **目标**：让你**摸到** Agent 的样子。LLM 是 mock 的，循环是真的。
> 跑完这个，下一章再看真实 `query.ts` 就有体感了。

---

## 🐍 完整代码（30 行可运行）

```python
# mini_agent_demo.py
# ============================================================
# 这是一个"假"的 Agent：LLM 部分是 mock 的，循环结构是真的。
# 目的：让你看清 Agent = LLM + Tools + Loop 的最小形态。
# ============================================================

# ---------- 1. 定义一个工具（普通的 Python 函数）----------
def tool_read_file(path: str) -> str:
    """读文件并返回内容。失败就返回错误信息。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"

# 工具注册表：名字 -> 函数。Agent 通过这个表查工具。
TOOLS = {"read_file": tool_read_file}

# ---------- 2. 假装的 LLM（实际项目里会换成 Claude API）----------
def fake_llm(messages: list) -> dict:
    """模拟一个会调工具的 LLM。"""
    # 数对话里有几条 "tool_result"，作为"轮次"判断
    tool_results_seen = sum(1 for m in messages if m.get("role") == "tool")
    if tool_results_seen == 0:
        # 第一轮：要调工具
        return {
            "stop_reason": "tool_use",
            "tool_name": "read_file",
            "tool_args": {"path": "README.md"}
        }
    else:
        # 第二轮：拿到结果后总结
        return {
            "stop_reason": "end_turn",
            "text": "我读完 README 了，看起来是个 Python 项目。"
        }

# ---------- 3. Agent 主循环（核心的 15 行）----------
def run_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    while True:                                      # ⭐ 无限循环
        response = fake_llm(messages)                # 调 LLM

        if response["stop_reason"] == "end_turn":
            print("Assistant:", response["text"])
            break                                    # 退出 while

        if response["stop_reason"] == "tool_use":
            name = response["tool_name"]
            args = response["tool_args"]
            print(f"[Agent 决定调用工具] {name}({args})")
            result = TOOLS[name](**args)              # 执行工具
            messages.append({"role": "tool", "name": name, "content": result[:200]})

# ---------- 4. 入口 ----------
if __name__ == "__main__":
    run_agent("帮我看看这个项目")
```

---

## 🔧 运行

```bash
cd ~/llm-study   # 任何有 README.md 的目录
python3 mini_agent_demo.py
```

预期输出：
```
[Agent 决定调用工具] read_file({'path': 'README.md'})
Assistant: 我读完 README 了，看起来是个 Python 项目。
```

恭喜，你跑了你的**第一个 Agent**——虽然 LLM 是假的，但循环和工具调用是真的。

---

## 🐍 Python 语法速记（针对初学者）

| 写法 | 含义 |
|---|---|
| `def f(x: str) -> str:` | 类型注解。`x: str` 提示 x 是字符串，`-> str` 提示返回字符串。**只是提示，不强制** |
| `with open(...) as f:` | 上下文管理器。块结束时自动关文件，比 `try/finally` 简洁 |
| `dict.get("k")` | 字典查 key，没有就返回 `None`，**不报错** |
| `**args` | 字典展开成关键字参数。`f(**{"a":1})` 等价 `f(a=1)` |
| `while True: ... break` | Python 写无限循环 + 条件退出的惯用法 |
| `if __name__ == "__main__":` | "只有直接运行这个文件时才执行下面"。被 import 时不执行 |
| `sum(1 for m in xs if cond)` | 生成器表达式 + sum，等价"满足条件的元素个数" |

---

## ⭐ 加深理解：对照看 5 步

把 [上一节](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md) 的 5 步标注到代码上：

```python
def run_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]  # 步骤 1：放进 messages
    while True:
        response = fake_llm(messages)                      # 步骤 2：调 LLM

        # ─── 步骤 3：看 stop_reason ───
        if response["stop_reason"] == "end_turn":
            print("Assistant:", response["text"])
            break                                          # → 结束
        if response["stop_reason"] == "tool_use":
            # ─── 步骤 4：执行工具 + 把 result 塞回 messages ───
            result = TOOLS[response["tool_name"]](**response["tool_args"])
            messages.append({"role": "tool", "name": ..., "content": result})
            # ─── 步骤 5：循环继续（while True 隐式）───
```

**真实 Claude Code 的 `query.ts` 第 307 行也是 `while (true)`**——架构一样，只是工业级版本多了 1700 行的护栏。

---

## 🛠 动手练习

1. **修改 `fake_llm`**：让它在第二轮再调一次工具（比如再读 `package.json`），第三轮才结束。观察循环跑了几轮。
2. **加一个工具**：写一个 `tool_list_dir(path)` 返回某目录下的文件列表，注册到 `TOOLS`。
3. **观察消息列表**：在循环最后打印 `messages`，看每一轮它怎么变长。

完成后进入 [第 2 章](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)，看 Claude Code 真正的主循环长啥样。

---

## 🔗 延伸阅读

- 上一节：[03-最简核心循环](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md) —— 5 步的概念
- 下一章：[00-章节总览](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) —— 工业级实现
- 完整 MiniCC：[00-章节总览](../08-%E7%AC%AC8%E7%AB%A0-%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0MiniCC/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md) —— 把 LLM 换成真 Claude

---

⬅ [03-最简核心循环](03-%E6%9C%80%E7%AE%80%E6%A0%B8%E5%BF%83%E5%BE%AA%E7%8E%AF.md)　|	➡ [第 2 章 →](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
