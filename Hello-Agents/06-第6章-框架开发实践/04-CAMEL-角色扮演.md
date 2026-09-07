---
tags: [Hello-Agents, 第6章, CAMEL, 角色扮演, 引导性提示]
chapter: 6
section: 6.4
---

# 6.4 CAMEL —— 角色扮演 + 自主协作

⬅ [03-AgentScope-消息驱动](03-AgentScope-%E6%B6%88%E6%81%AF%E9%A9%B1%E5%8A%A8.md)	|	➡ [05-LangGraph-图驱动](05-LangGraph-%E5%9B%BE%E9%A9%B1%E5%8A%A8.md)

> 探索"**最少人类干预**下,让两个 Agent 通过**角色扮演自主协作**" 的开源框架。
> **极简理念**:**两个 Agent + 引导性提示 = 自主完成复杂任务**。

---

## 🎬 故事比喻:跨领域专家结对

想象:
- **股票交易员**(懂市场,不懂编程)
- **Python 程序员**(懂编程,不懂股票)

→ 让他们**结对工作**几小时,能造出一个**股票分析工具**!
→ CAMEL 就在做这件事 —— **让两个 AI 跨领域专家自主对话协作**。

---

## 🎯 6.4.1 CAMEL 的两大核心概念

### 概念 1:**角色扮演(Role-Playing)**

> **两个 Agent + 互补的明确角色**

```
任务: 开发股票交易策略分析工具

AI User      = 资深股票交易员
              (懂市场/策略,不懂编程)

AI Assistant = 优秀的 Python 程序员
              (精通编程,不懂股票)

协作方式: 跨领域专家对话
   - 交易员提需求 → 程序员实现
   - 完成单一 AI 都无法完成的复杂任务
```

→ **这种设定把任务转化为"跨领域专家对话"**。

### 概念 2:**引导性提示(Inception Prompting)**

> CAMEL **最核心的技术**。

**没有这个技术**:两个 AI 容易跑偏 / 陷入无意义对话。
**有了这个技术**:**AI 在没有人类监督的情况下保持在角色里、朝目标推进**。

#### Inception Prompting 包含 4 个关键部分

| 部分 | 例子 |
|---|---|
| **明确自身角色** | "你是一位资深股票交易员..." |
| **告知协作者角色** | "你正在与一位优秀的 Python 程序员合作..." |
| **定义共同目标** | "你们的共同目标是开发股票分析工具..." |
| **行为约束 + 沟通协议** | "AI User 一次只提出一个具体步骤" / "AI Assistant 完成上一步前不要追问" / "用 `<SOLUTION>` 标识任务完成" |

→ **第 4 项是关键** —— **规则保证对话不偏不乱**。

---

## 🆚 6.4.2 CAMEL vs AutoGen vs AgentScope

| | **CAMEL** | **AutoGen** | **AgentScope** |
|---|---|---|---|
| Agent 数 | **固定 2 个** | N 个 | N 个 |
| 协作模式 | 跨领域对话 | 群聊 | 消息驱动 |
| 复杂度 | **最低** | 中 | 高 |
| 应用场景 | 探索性研究 | 流程化协作 | 生产系统 |
| 学习曲线 | **平** | 中 | 陡 |
| 设计哲学 | **极简** | 灵活 | 工程化 |

→ CAMEL 的**精髓在于"少即是多"**。

---

## 🛠 6.4.3 实战:AI 心理学家 × AI 作家 写电子书

### 任务设定

让两位 AI 合作创作:
> **"拖延症心理学"短篇科普电子书**(8000-10000 字)

### 智能体角色

| Agent | 角色 | 专长 |
|---|---|---|
| **AI User** | **作家** | 写作技巧 + 叙述能力 + 读者体验 |
| **AI Assistant** | **心理学家** | 心理学理论 + 认知行为科学 + 实证研究 |

→ **跨领域协作**:作家**懂表达**,心理学家**懂内容**,两者结合 = 高质量科普。

### 关键代码

#### Step 1:**定义任务**

```python
from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# 创建模型(以 Qwen 为例)
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=LLM_MODEL,
    url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# 定义协作任务
task_prompt = """
创作一本关于"拖延症心理学"的短篇电子书,目标读者是对心理学感兴趣的普通大众。

要求:
1. 内容科学严谨,基于实证研究
2. 语言通俗易懂,避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在 8000-10000 字
5. 结构清晰,包含引言、核心章节和总结
"""
```

→ **`task_prompt` 是整个协作的"任务说明书"**,CAMEL 会用它生成两个 Agent 的 Inception Prompting。

#### Step 2:**初始化 RolePlaying 会话**

```python
role_play_session = RolePlaying(
    assistant_role_name="心理学家",         # AI Assistant
    user_role_name="作家",                  # AI User
    task_prompt=task_prompt,
    assistant_agent_kwargs={"model": model},
    user_agent_kwargs={"model": model},
    chat_turn_limit=10,                      # 最大对话轮次
)
```

→ **关键设计**:
- **`user` = 推动者 + 需求方** → 作家(规划结构、提出要求)
- **`assistant` = 执行者 + 方案提供方** → 心理学家(填充专业内容)

#### Step 3:**驱动自动化对话**

```python
print(f"协作任务:\n{task_prompt}\n")

# 启动对话
input_msg = role_play_session.init_chat()

n = 0
while n < role_play_session.chat_turn_limit:
    n += 1

    # 双方各回复一次
    assistant_response, user_response = role_play_session.step(input_msg)

    # 检测终止
    if assistant_response.terminated or user_response.terminated:
        print("对话因终止条件结束")
        break

    print_text_animated(f"\nAI 作家:\n{user_response.msg.content}\n")
    print_text_animated(f"\nAI 心理学家:\n{assistant_response.msg.content}\n")

    # 检测任务完成
    if "<CAMEL_TASK_DONE>" in user_response.msg.content:
        break

    # 助理的回复作为下一轮 user 的输入
    input_msg = assistant_response.msg
```

→ **就这点代码,两个 AI 自主对话直到完成任务**。

---

## 🎮 6.4.4 真实对话片段(片段示意)

```
AI 作家:
为了创作这本电子书,我建议按以下结构进行:
1. 引言(800 字):介绍拖延症的普遍性,引出核心问题
2. 第一章:什么是拖延症(1500 字)- 需要你提供学术定义和分类
3. 第二章:拖延症的心理机制(2000 字)- 详细的认知行为科学解读
4. 第三章:案例分析(1500 字)
5. 第四章:实用改善建议(2000 字)
6. 总结(500 字)

请先帮我写第一章的核心内容,要求包含主流心理学定义、3 种分类、相关研究。

AI 心理学家:
好的,以下是第一章的核心内容...

【一、什么是拖延症】

拖延症(Procrastination)是一种自我调节的失败,...

主流定义:Steel (2007) 将拖延定义为"自愿延迟原定计划的行动,
尽管知道这种延迟会带来不良后果"。
[研究引用:Steel, P. (2007). The nature of procrastination: A meta-analytic and
theoretical review of quintessential self-regulatory failure. Psychological Bulletin.]

三种分类:
1. 唤醒型拖延(Arousal Procrastination)- 享受最后期限的刺激感
2. 回避型拖延(Avoidance Procrastination)- 害怕失败或评价
3. 决策型拖延(Decisional Procrastination)- 面对选择时的瘫痪
...

AI 作家:
非常好,这章内容专业且有引用!
现在请帮我把这部分内容包装成读者友好的语言,加入一个真实案例(可以是虚构但符合实际)...

[对话继续...]
```

→ **AI 作家不断要求,AI 心理学家不断提供专业内容** → **一本科普电子书的雏形**。

---

## ⭐ 6.4.5 CAMEL 的优势与局限

### ✅ 优势

| 优势 | 解释 |
|---|---|
| **极简** | 2 个 Agent,几行代码搞定 |
| **跨领域协作天然适配** | 设计就是为了"两个专家结对" |
| **自主性强** | Inception Prompting 让 AI 自己干 |
| **降低设计复杂度** | 不用设计复杂的状态机/流程 |
| **学术研究友好** | 论文出处明确,易于复现 |

### ⚠️ 局限

| 局限 | 解释 |
|---|---|
| **只支持 2 个 Agent** | 复杂协作(>2 个角色)不擅长 |
| **依赖 LLM 质量** | 弱模型 → Inception Prompting 失效 |
| **流程不可控** | 不像 AutoGen RoundRobin 那样明确 |
| **生产化能力弱** | 主要面向研究,不是企业级 |
| **生态相对小** | 不如 LangChain/AutoGen |

---

## 🎯 6.4.6 CAMEL 的独特价值

### 价值 1:**跨领域协作的最佳实践**

适合场景:
- 程序员 + 产品经理(技术方案 + 需求)
- 医生 + 律师(医疗 + 法律咨询)
- 作家 + 编辑(创作 + 修改)
- 老师 + 学生(教 + 学)

### 价值 2:**学术研究探索**

CAMEL 的论文很有名:
> _CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society_

它**催生了"AI 社会模拟"研究方向**。
做学术研究的话,**CAMEL 是首选**。

### 价值 3:**学习多 Agent 设计哲学**

```
"双 Agent + 明确角色 + Inception Prompting"
   ↓
是多 Agent 系统的 "Hello World"
学透了这个,理解其他多 Agent 框架就容易
```

---

## ⚠️ 小白避坑

1. **不要试图加 > 2 个 Agent**
   - CAMEL 不是为多 Agent 设计的
   - 真要多 Agent 用 AutoGen / AgentScope
2. **角色要"互补"而非"相似"**
   - "两个程序员" 不如 "程序员 + 设计师"
3. **task_prompt 越具体越好**
   - 太模糊 → Inception Prompting 不知道怎么生成
4. **`chat_turn_limit` 必须设**
   - 防止两个 AI 互相吹捧不停
5. **`<CAMEL_TASK_DONE>` 是 CAMEL 的"接力暗号"**
   - 记住这个终止信号

---

## 📊 CAMEL 与其他框架对比

| 维度 | **CAMEL** | AutoGen | AgentScope | LangGraph |
|---|---|---|---|---|
| Agent 数 | **2** | N | N | N |
| 哲学 | **角色扮演** | 对话 | 消息驱动 | 图驱动 |
| 适合 | **跨领域协作** | 团队协作 | 大规模生产 | 复杂工作流 |
| 复杂度 | **最低** | 中 | 高 | 中 |
| 学习曲线 | **最平** | 中 | 陡 | 中 |
| 学术受欢迎 | **高** | 中 | 中 | 中 |

---

## 🎯 6.4.7 何时选 CAMEL?

```
✅ 适合:
- 学习多 Agent 设计入门
- 跨领域协作探索(如医疗 + 法律)
- 学术研究 + 论文复现
- 写电子书 / 创作类协作任务
- 不想写复杂代码

❌ 不适合:
- 多于 2 个 Agent 的复杂协作(用 AutoGen)
- 生产级应用(用 AgentScope)
- 需要复杂控制流(用 LangGraph)
- 需要工具调用密集的场景
```

---

## 📌 6.4 节要点

| 知识点 | 一句话 |
|---|---|
| **核心哲学** | 角色扮演 + 引导性提示 |
| **架构** | **2 个 Agent**(AI User + AI Assistant) |
| **Inception Prompting** | 4 部分:自身角色 + 协作者角色 + 共同目标 + **行为约束** |
| **极简设计** | 几行代码 → 自主对话协作 |
| **典型应用** | 跨领域协作 / 学术研究 |
| **致命短板** | 不支持 > 2 个 Agent |

---

## 🔗 延伸阅读

- 上一节:[03-AgentScope-消息驱动](03-AgentScope-%E6%B6%88%E6%81%AF%E9%A9%B1%E5%8A%A8.md)
- 下一节:[05-LangGraph-图驱动](05-LangGraph-%E5%9B%BE%E9%A9%B1%E5%8A%A8.md) —— 最灵活的控制流
- 论文:CAMEL: Communicative Agents for "Mind" Exploration of LLM Society
- 官方:https://www.camel-ai.org/

---

⬅ [03-AgentScope-消息驱动](03-AgentScope-%E6%B6%88%E6%81%AF%E9%A9%B1%E5%8A%A8.md)	|	➡ [05-LangGraph-图驱动](05-LangGraph-%E5%9B%BE%E9%A9%B1%E5%8A%A8.md)
