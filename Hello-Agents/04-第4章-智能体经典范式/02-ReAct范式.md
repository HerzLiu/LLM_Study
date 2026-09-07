---
tags: [Hello-Agents, 第4章, ReAct, 范式, 实战]
chapter: 4
section: 4.2
---

# 4.2 ReAct 范式(完整实现)

⬅ [01-环境与LLM客户端封装](01-%E7%8E%AF%E5%A2%83%E4%B8%8ELLM%E5%AE%A2%E6%88%B7%E7%AB%AF%E5%B0%81%E8%A3%85.md)	|	➡ [03-Plan-and-Solve范式](03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md)

> ⭐ **第一个真正的 Agent 范式**:**Re**asoning + **Act**ing 紧密结合的"边想边做"。
> Shunyu Yao 于 2022 年提出 [1]。

---

## 🎬 故事比喻:侦探办案

ReAct 像一个**侦探**:
- **思考(Thought)**:"我现在掌握什么线索?下一步该查啥?"
- **行动(Action)**:"去现场提取指纹"
- **观察(Observation)**:"指纹和嫌疑人 A 匹配"
- **下一轮思考**:"那我下一步该审问 A"

→ **边查边推理,根据线索动态调整方向**。

---

## 🆚 4.2.1 ReAct 之前的两个对比派

ReAct 出现之前,有两种主流方法:

| 方法 | 特点 | 缺陷 |
|---|---|---|
| **纯思考**(CoT 思维链) | LLM 自己推理 | ❌ 无法访问外部世界,**幻觉严重** |
| **纯行动**(直接调工具) | LLM 直接输出动作 | ❌ 缺乏规划和纠错能力 |

### ReAct 的革命性洞察

> **思考 + 行动 = 1+1 > 2**
> - 思考**指导行动**(避免乱调工具)
> - 行动结果**修正思考**(避免幻觉)

→ 这就是 [第 1 章](../01-%E7%AC%AC1%E7%AB%A0-%E5%88%9D%E8%AF%86%E6%99%BA%E8%83%BD%E4%BD%93/02-%E6%9E%84%E6%88%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E5%8E%9F%E7%90%86.md) 讲的 **T-A-O 范式**的理论基础。

### 形式化表达

在第 t 步,LLM 根据初始问题 q 和历史 H_t = (T_1, A_1, O_1, ..., T_{t-1}, A_{t-1}, O_{t-1}),生成新的思考 T_t 和行动 A_t:

```
(T_t, A_t) = LLM(q, H_t)
```

然后环境工具执行 A_t 返回观察 O_t:

```
O_t = Tool(A_t)
```

循环直到 LLM 在 T_t 中判定任务完成。

---

## 🎯 4.2.2 实战目标

构建一个能**回答时效性问题**的 ReAct Agent:
> **"华为最新的手机是哪一款?它的主要卖点是什么?"**

这种问题 **LLM 自己答不了**(知识有截止日期),**必须搜网**。

---

## 🛠 4.2.3 定义工具(Tool)

### 工具的 3 要素

| 要素 | 例子 |
|---|---|
| **名称** | `Search` |
| **描述** | "用于查询时事、事实信息" |
| **执行逻辑** | 调用搜索 API 的函数 |

### 实现搜索工具(SerpApi)

```bash
pip install google-search-results
```

`.env` 加上:
```
SERPAPI_API_KEY="YOUR_KEY"
```

```python
import os
from serpapi import SerpApiClient

def search(query: str) -> str:
    """SerpApi 网页搜索 —— 智能解析返回最精确的答案"""
    print(f"🔍 正在执行搜索: {query}")
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "gl": "cn", "hl": "zh-cn",
        }
        results = SerpApiClient(params).get_dict()

        # ⭐ 智能解析:优先返回最精确的答案
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results:
            # 取前 3 条搜索结果摘要
            snippets = [
                f"[{i+1}] {r.get('title','')}\n{r.get('snippet','')}"
                for i, r in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f"搜索时发生错误: {e}"
```

### ⭐ 关键设计:智能解析

```python
if "answer_box" in results:
    return results["answer_box"]["answer"]   # 最精确
elif "knowledge_graph" in results:
    return results["knowledge_graph"]["description"]
else:
    # 退而求其次,返回搜索摘要
```

**为什么这样设计?**
- LLM 看到**精确答案**比看搜索摘要**更省 token + 更准**
- 自动**降级处理**,各种情况都能应对

---

## 📦 4.2.4 工具执行器(ToolExecutor)

多个工具需要统一管理:

```python
from typing import Dict, Any

class ToolExecutor:
    """工具管理器:注册 + 查找 + 列出"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """注册一个工具"""
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """根据名字取工具函数"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """格式化所有工具描述(给 LLM 看)"""
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
```

### 注册搜索工具

```python
tool_executor = ToolExecutor()
tool_executor.registerTool(
    "Search",
    "一个网页搜索引擎。当你需要回答关于时事、事实以及知识库外信息时使用。",
    search
)
```

---

## 🤖 4.2.5 ReAct Agent 完整实现

### Step 1:精心设计的 Prompt 模板

```python
REACT_PROMPT_TEMPLATE = """
请注意,你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:
Thought: 你的思考过程,用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动,必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`: 调用一个可用工具。
- `Finish[最终答案]`: 当你认为已经获得最终答案时。

当你收集到足够信息能够回答用户问题时,必须使用 Action: Finish[最终答案]。

现在,请开始解决以下问题:
Question: {question}
History: {history}
"""
```

### Step 2:核心循环

```python
import re
from llm_client import HelloAgentsLLM

class ReActAgent:
    def __init__(self, llm_client, tool_executor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps    # ⭐ 防死循环
        self.history = []

    def run(self, question: str):
        """主循环:思考 → 行动 → 观察 → 重复"""
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 拼 prompt
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )

            # 2. 调 LLM 思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                break

            # 3. 解析输出
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                break

            # 4. 判断是否结束
            if action.startswith("Finish"):
                final = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final}")
                return final

            # 5. 执行工具
            tool_name, tool_input = self._parse_action(action)
            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误: 未找到工具 '{tool_name}'"
            else:
                observation = tool_function(tool_input)

            print(f"👀 观察: {observation}")

            # 6. ⭐ 把 Action + Observation 加进历史
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数,流程终止。")
        return None

    def _parse_output(self, text: str):
        """从 LLM 输出中提取 Thought 和 Action"""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析 Action: Search[xxx] → ("Search", "xxx")"""
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
```

### 📝 核心代码深度解读

#### 🔑 关键 1:**`max_steps` 是安全阀**
```python
while current_step < self.max_steps:
```
→ **防止 LLM 陷入死循环**。一般设 5~10。

#### 🔑 关键 2:**正则解析输出**
```python
thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
```
- `\s*` —— 匹配任意空白
- `(.*?)` —— **非贪婪匹配**(到下一个标记就停)
- `(?=\nAction:|$)` —— **正向预查**:遇到 `\nAction:` 或文本末尾就停
- `re.DOTALL` —— 让 `.` 也匹配换行符

→ 这种"格式驱动"解析是 ReAct 工程化的关键。
→ 生产级用 **OpenAI Function Calling** 更稳。

#### 🔑 关键 3:**历史拼接逻辑**
```python
self.history.append(f"Action: {action}")
self.history.append(f"Observation: {observation}")
```
→ 注意:**只加 Action 和 Observation,不加 Thought**。
→ 因为 Thought 已经在上一次 LLM 输出里了。
→ 下次 LLM 看到完整历史会**自动续写新的 Thought**。

#### 🔑 关键 4:**Finish 退出机制**
```python
if action.startswith("Finish"):
    final = re.match(r"Finish\[(.*)\]", action).group(1)
    return final
```
→ **必须明示 LLM 怎么"收工"**,否则它会一直循环。

---

## 🎮 4.2.6 运行实例

### 真实运行结果

```
Question: 华为最新的手机是哪一款?它的主要卖点是什么?

--- 第 1 步 ---
🤔 思考: 要回答这个问题,我需要查找华为最新发布的手机型号及其主要特点。
        这些信息可能在我的现有知识库之外,因此需要使用搜索引擎。
🎬 行动: Search[华为最新手机型号及主要卖点]
🔍 正在执行搜索: 华为最新手机型号及主要卖点
👀 观察: [1] 华为手机- 华为官网
         智能手机 ; Mate 系列. 非凡旗舰 · HUAWEI Mate XTs. 非凡大师...
         [2] 2025年华为手机哪一款性价比高...HUAWEI Mate 70...
         [3] HUAWEI Pura 80 Pro+ ...先锋影像...

--- 第 2 步 ---
🤔 思考: 根据搜索结果,华为最新旗舰是 Mate 70 和 Pura 80 Pro+。
        我可以提取它们的主要卖点。
🎉 最终答案: 华为的最新手机是 HUAWEI Pura 80 Pro+ 或 HUAWEI Mate 70。
            Mate 70 主打顶级拍照配置;Pura 80 Pro+ 强调先锋影像技术。
```

→ **2 步搞定**!这就是 ReAct 处理时效性问题的威力。

---

## ⭐ 4.2.7 ReAct 的特点与局限

### ✅ 优点

| 优点 | 解释 |
|---|---|
| **高可解释性** | Thought 链让你看到 LLM "心路历程" |
| **动态规划与纠错** | "走一步看一步",根据 Observation 调整 |
| **工具协同** | LLM 推理 + 工具执行,**优势互补** |

### ⚠️ 局限

| 局限 | 解释 |
|---|---|
| **强依赖 LLM 能力** | 弱模型 → Thought 错乱,格式不遵循 |
| **执行效率低** | 每步都调 LLM,**串行 + 多次调用** |
| **提示词脆弱** | 模板改一个字就可能崩 |
| **可能陷入局部最优** | 缺乏全局规划,可能**原地打转** |

→ 局限 4 正是 **Plan-and-Solve** 范式要解决的问题(下一节)。

---

## 🔧 4.2.8 调试技巧(实战必备)

| 技巧                           | 怎么做                                             |
| ---------------------------- | ----------------------------------------------- |
| **打印完整 prompt**              | 每次调 LLM 前 `print(prompt)`,看到底喂了什么               |
| **打印原始 LLM 输出**              | 解析失败时打印原始文本,**判断是 LLM 错还是解析逻辑错**                |
| **验证工具输入输出**                 | 检查 `tool_input` 格式 + `observation` 是 LLM 能理解的形式 |
| **加 few-shot 示例**            | 模板里塞 1-2 个完整 T-A-O 案例,**示范效果远超描述**              |
| **换更强的模型 / 调 temperature=0** | 直接解决格式不遵循问题                                     |

---

## ⚠️ 小白避坑

1. **不要用基础模型(未指令调优)做 ReAct**
   - 基础模型不会"听话",格式遵循极差
   - 必须用 Chat / Instruct 版本
2. **`max_steps` 别太大**
   - 一般 5-10 够用
   - **死循环的 Agent 烧钱速度惊人**
3. **Observation 太长会浪费 token**
   - 加摘要(让 LLM 生成 Observation 摘要)
4. **生产用 Function Calling**
   - 比正则解析稳定 10 倍
   - OpenAI / Claude / 国内主流 LLM 都支持

---

## 📌 4.2 节要点

| 知识点 | 一句话 |
|---|---|
| **ReAct 核心** | Thought + Action + Observation 循环 |
| **工具 3 要素** | 名称 / 描述 / 执行函数 |
| **Prompt 设计** | 工具列表 + 格式约束 + Finish 机制 |
| **`max_steps`** | 防死循环的安全阀 |
| **正则解析** | 学习用,生产用 Function Calling |
| **优势** | 可解释 + 动态纠错 + 工具协同 |
| **局限** | 慢 + 提示脆弱 + 缺全局规划 |

---

## 🔗 延伸阅读

- 上一节:[01-环境与LLM客户端封装](01-%E7%8E%AF%E5%A2%83%E4%B8%8ELLM%E5%AE%A2%E6%88%B7%E7%AB%AF%E5%B0%81%E8%A3%85.md)
- 下一节:[03-Plan-and-Solve范式](03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md) —— 弥补 ReAct 的"缺全局规划"
- 第 1 章实战:[第 1 章简化版 ReAct](../01-%E7%AC%AC1%E7%AB%A0-%E5%88%9D%E8%AF%86%E6%99%BA%E8%83%BD%E4%BD%93/03-%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0%E6%97%85%E8%A1%8C%E5%8A%A9%E6%89%8B.md)
- Function Calling 进阶:[Happy-LLM Tiny-Agent](../../Happy-LLM/07-%E7%AC%AC7%E7%AB%A0-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8/05-Tiny-Agent%E5%AE%9E%E6%88%98%E9%80%9F%E8%A7%88.md)

---

⬅ [01-环境与LLM客户端封装](01-%E7%8E%AF%E5%A2%83%E4%B8%8ELLM%E5%AE%A2%E6%88%B7%E7%AB%AF%E5%B0%81%E8%A3%85.md)	|	➡ [03-Plan-and-Solve范式](03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md)
