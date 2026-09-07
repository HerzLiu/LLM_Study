---
tags: [Claude-Code, 第3章, API协议, Tool-Use]
chapter: 3
section: 3.1
---

# 3.1 Tool Use 协议基础（Anthropic API）

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-Tool接口设计](02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md)

> ⭐ Claude Code 工具系统建立在这套协议之上。先理解协议本身。

---

## 🎬 故事比喻：服务员的标准化下单

厨师（LLM）能开口说话，但他要喊得让传菜员（你的程序）能听懂。
**Tool Use 协议**就是一套强制的"喊话格式"：
- 名字标准（不能喊"那个白色颗粒"，要喊"盐"）
- 参数清楚（不能"加一点"，要"加 30 克"）
- 反馈格式（盐拿回来后要说"30 克盐已到 1 号台"）

---

## 🔧 三段协议

### Step 1：发请求时——把"工具说明书"一起发

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=[                                  # ⭐ 工具说明书
        {
            "name": "read_file",             # LLM 喊话用这个名字
            "description": "Read a file from local filesystem.",
            "input_schema": {                # JSON Schema 格式
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path"
                    }
                },
                "required": ["path"]
            }
        }
    ],
    messages=[{"role": "user", "content": "看看 /etc/hosts"}]
)
```

### Step 2：收响应时——content 是"块列表"

```python
# response.content 可能长这样
[
    {"type": "text", "text": "好的，我来读一下"},   # 文本块
    {                                                # 工具调用块
        "type": "tool_use",
        "id": "toolu_01abc",                         # 唯一 ID
        "name": "read_file",
        "input": {"path": "/etc/hosts"}              # 模型填的参数
    }
]
# response.stop_reason == "tool_use"
```

**注意**：`content` 不是字符串，是**块的数组**。模型可以同时输出文本和多个工具调用。

### Step 3：执行完工具——用 `tool_result` 塞回

```python
# 把 assistant 完整回复存下
messages.append({"role": "assistant", "content": response.content})

# 再加一条 user 角色的 tool_result
messages.append({
    "role": "user",                              # ⚠️ 注意是 user！
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01abc",        # 对应上面 tool_use 的 id
            "content": "127.0.0.1 localhost\n..."
        }
    ]
})
```

---

## ⚠️ 协议陷阱：tool_result 是 user 角色！

最容易踩的坑：**`tool_result` 是用 `role: "user"` 发回去的，不是 `role: "tool"`。**

| 框架 | tool_result 的 role |
|---|---|
| Anthropic API | `user` |
| OpenAI API | `tool` |

**为什么 Anthropic 这么设计？**
> 把"工具结果"看作"用户的反馈"——你（程序）作为用户的代理在帮模型获取信息。

记错这一点的话，API 会直接报错。

---

## ⭐ 加深理解：循环里 messages 是怎么长的

每完整一轮 tool call，messages 增加 2 条：

```python
# 初始
[
    {"role": "user", "content": "看看 /etc/hosts"}
]

# 调一次 API 后
[
    {"role": "user", "content": "看看 /etc/hosts"},
    {"role": "assistant", "content": [                  # +1 条
        {"type": "text", "text": "好的，我来读一下"},
        {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {...}}
    ]},
    {"role": "user", "content": [                       # +1 条（注意是 user！）
        {"type": "tool_result", "tool_use_id": "tu1", "content": "..."}
    ]}
]

# 再调 API → 可能又 +2 条
```

**循环的本质就是这条 messages 在不停伸长**。
不停伸长就会爆 → 触发 auto-compact（详见 [第 4 章](../04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)）。

---

## 🐍 完整可运行 demo（需 API key）

```python
# tool_use_demo.py
import anthropic
import json
import os

client = anthropic.Anthropic()

# 1. 定义工具
TOOLS = [{
    "name": "list_dir",
    "description": "List files in a directory",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    }
}]

# 2. 工具的真实实现
def execute_tool(name, args):
    if name == "list_dir":
        return "\n".join(os.listdir(args["path"]))[:500]
    return "Unknown tool"

# 3. 完整循环
def run(user_input):
    messages = [{"role": "user", "content": user_input}]
    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )

        # 找 tool_use 块
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b for b in resp.content if b.type == "text"]
        for tb in text_blocks:
            print("Claude:", tb.text)

        # 存 assistant 回复
        messages.append({"role": "assistant", "content": resp.content})

        if not tool_uses:
            return                                       # 结束

        # 执行所有工具
        tool_results = []
        for tu in tool_uses:
            print(f"  [tool] {tu.name}({tu.input})")
            result = execute_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result
            })

        # 塞回（注意 user 角色！）
        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    run("看看当前目录有什么文件")
```

跑这个需要 `export ANTHROPIC_API_KEY=...`。

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-Tool接口设计](02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md) —— CC 在协议之上加了什么
- Anthropic 官方文档：https://docs.claude.com/en/docs/build-with-claude/tool-use

---

## ❓ 小测验

> 1. `tools` 字段里每个工具有哪 3 个必备字段？
> 2. `tool_result` 是用什么 `role` 发回去的？
> 3. 一轮完整的工具调用，messages 数组增加几条？

→ 答案见 [自测题汇总](../%E9%99%84%E5%BD%95/%E8%87%AA%E6%B5%8B%E9%A2%98%E6%B1%87%E6%80%BB.md#31-tool-use-%E5%8D%8F%E8%AE%AE)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-Tool接口设计](02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md)
