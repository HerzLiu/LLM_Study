---
tags: [Hello-Agents, 第5章, n8n, 工作流自动化, 低代码]
chapter: 5
section: 5.4
---

# 5.4 n8n —— 通用自动化 + AI

⬅ [03-Dify-开源全栈平台](03-Dify-%E5%BC%80%E6%BA%90%E5%85%A8%E6%A0%88%E5%B9%B3%E5%8F%B0.md)	|	➡ [05-三大平台对比与选型](05-%E4%B8%89%E5%A4%A7%E5%B9%B3%E5%8F%B0%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)

> **不是 LLM 平台,是工作流自动化平台**。LLM 只是它**几百个节点中的一个**。
> 适合**把 AI 嵌入现有业务流程**。

---

## 🎬 故事比喻:万能管线工

- **Coze** 像专做"AI 玩具"的工厂
- **Dify** 像专做"AI 应用"的全栈工厂
- **n8n** 像**万能管线工** —— 把你家所有水管/电线/网络**连起来**

n8n 的口号:**"Connect anything to everything"**(连接万物到万物)。
**AI 只是连接的对象之一**。

---

## 🎯 5.4.1 n8n 是什么?

| 维度        | 描述                                |
| --------- | --------------------------------- |
| **核心身份**  | **通用工作流自动化平台**(非 LLM 专属)          |
| **关键能力**  | 数百个预置节点(SaaS / API / 数据库)         |
| **AI 能力** | **AI Agent 节点**(整合 LLM + 工具 + 记忆) |
| **部署方式**  | 本地 / 云端 / Docker                  |
| **适合人群**  | 需要将 AI 嵌入**现有业务流程**的开发者           |

### ⭐ 核心理念差异

```
Coze / Dify:
   主体是 AI Agent,工具是辅助

n8n:
   主体是 工作流(Workflow),AI 是其中一个节点
```

---

## 🧱 5.4.2 核心概念:节点 + 工作流

### 节点(Node)= 积木块

| 类型 | 描述 |
|---|---|
| **触发节点**(Trigger) | 启动工作流(每个 workflow **必须有且只有一个**) |
| **常规节点**(Regular) | 处理数据 / 执行逻辑 |

**触发节点例子**:
- "收到一封 Gmail 邮件时"
- "每小时定时触发"
- "收到 Webhook 请求时"

**常规节点例子**:
- "读 Google Sheets"
- "调 OpenAI API"
- "写入 PostgreSQL"

→ n8n 提供**数百个预置节点**,几乎所有主流 SaaS/数据库/API 都覆盖。

### 工作流(Workflow)= 节点连成的流程图

- 节点之间用线连接
- 数据以 **JSON** 格式在节点间流动
- 可视化定义"数据如何流动 → 如何被处理 → 最终完成什么"

→ **n8n 的真正威力 = 连接**。

---

## 🛠 5.4.3 实战:智能邮件助手(集成 RAG + 工具)

### 案例目标

构建一个**智能邮件 Agent**:
- 自动接收 Gmail
- AI 分析意图
- 调用工具(搜索 / 查私有知识库)
- 智能生成回复
- 自动发送

### 架构

```
┌────────────────────────────────────────┐
│  接收 Gmail (触发)                       │
└──────────────┬──────────────────────────┘
               │
               ▼
       ┌───────────────────┐
       │   AI Agent 节点    │
       │   ┌───────────┐    │
       │   │ LLM       │    │
       │   │ Memory    │    │
       │   │ Tools     │    │
       │   └───────────┘    │
       └───────┬───────────┘
               │
               ▼
       ┌───────────────┐
       │ 发送 Gmail 回复 │
       └───────────────┘

工具:
- SerpAPI (上网搜索)
- Simple Vector Store (私有 RAG 知识库)
```

### ⭐ n8n 的设计精妙处

> 传统方式需要把工具拆成多个子工作流,**n8n 的 AI Agent 节点把 LLM + 记忆 + 工具整合在一个统一界面**。

→ **极大简化构建过程**。

---

### 🛠 Step 1:准备 RAG 私有知识库(独立工作流)

**目标**:让 Agent 能回答关于"我"的私有信息(工作时间/邮件回复策略)。

#### (1) **定义知识源**(Code 节点)

```javascript
return [
  {
    "doc_id": "work-schedule-001",
    "content": "我的工作时间是周一至周五,上午9点到下午5点。时区是 AEST。"
  },
  {
    "doc_id": "off-hours-policy-001",
    "content": "在非工作时间(包括周末和公共假期),我无法立即回复邮件。"
  },
  {
    "doc_id": "auto-reply-instruction-001",
    "content": "如果邮件是非工作时间收到的,AI 助手应告知发件人,
              我会在下一个工作日 9-5 点之间尽快处理。"
  }
];
```

#### (2) **文本向量化**(Embeddings 节点)

- 节点:`Embeddings Google Gemini`
- 模型:`gemini-embedding-exp-03-07`
- 作用:把文本翻译成 LLM 能"理解"的向量

#### (3) **存入向量库**(Simple Vector Store)

```yaml
节点: Simple Vector Store
配置:
  Operation Mode: Insert Documents
  Memory Key: my-dailytime    # 知识库的"表名"
```

→ 手动执行一次,知识就**加载到 n8n 内存**了。

---

### 🛠 Step 2:Agent 主工作流

#### (1) **Gmail 触发器**

```yaml
节点: Gmail
Event: Message Received
作用: 每当新邮件进入,自动触发
```

⚠️ Gmail API 需要在 Google Cloud Console 配置 OAuth 凭证 + 回调 URL。

#### (2) **AI Agent 节点**(核心)

```yaml
节点: AI Agent

Chat Model:
  - Google Gemini Chat Model    # "大脑"

Memory:
  - Simple Memory               # 多轮对话记忆
  - Key: {{ $('Gmail').item.json.threadId }}   # 用邮件线程 ID 作唯一标识

Tools:
  - SerpAPI                     # 上网搜索
  - Simple Vector Store         # 私有知识库
```

→ **3 件套整合在一个节点里**:LLM + Memory + Tools。

#### (3) **关键:Prompt 设计**

```markdown
# System Message

你是全天候待命、专业高效的 AI 邮件助手。
第一时间使用公开信息尽力回答所有邮件问题,
并根据我的工作日程,在回复开头附加上下文状态提醒。

## 上下文
- 当前时间: {{ new Date().toLocaleString('en-AU') }}
- 邮件信息在输入数据中

## 可用工具
- Simple Vector Store2: 查询我准确的工作时间
- SerpAPI: 互联网搜索回答邮件问题

## 执行步骤
1. **分析问题**: 提炼发件人核心问题
2. **并行搜集**:
   a. SerpAPI 搜索答案
   b. Simple Vector Store2 获取工作时间
3. **草拟回复**: 基于 SerpAPI 结果回答
4. **添加状态前缀**:
   - 非工作时间 → 加礼貌提醒 + 我的工作时间
   - 工作时间 → 简单问候即可
5. **输出 JSON**:
   {
     "shouldReply": true,
     "subject": "Re: [原始主题]",
     "body": "[完整回复, 换行用 <br>]"
   }

## 规则
- 永远优先尝试回答
- 必须声明状态
- 信息来源准确(工作时间从 Vector Store,问题答案从 SerpAPI)
- 不编造信息
```

→ 这就是**真正的 ReAct + RAG + 工具调用** 一气呵成。

#### (4) **Simple Vector Store 工具配置**(关键 3 项必须一致)

```yaml
Operation Mode: Retrieve Documents (As Tool for AI Agent)
Memory Key: my_private_knowledge         # ⭐ 必须和 Step 1 完全一致
Embeddings: Google Gemini (同一模型)    # ⭐ 必须用同一模型
Description: "查询我的个人信息,特别是工作时间和邮件策略"
```

⚠️ **三项不一致 → Agent 取不到知识**(初学者常踩这个坑)

#### (5) **发送回复**(Gmail Send 节点)

```yaml
To: {{ $('Gmail').item.json.From }}
Subject: Re: {{ $('Gmail').item.json.Subject }}
Message: {{ $json.output }}   # 来自 AI Agent 的输出
```

→ **闭环完成**。

---

## 🎮 5.4.4 真实运行效果

| 时间        | 用户邮件              | Agent 自动回复                                                  |
| --------- | ----------------- | ----------------------------------------------------------- |
| 工作日 14:00 | "Python 中如何实现快排?" | 直接回答 + 代码示例                                                 |
| 周日 22:00  | "如何用 RAG 解决幻觉?"   | "您好,您已在我的非工作时间联系我(我的工作时间为周一至周五 9-5)。我会在下一个工作日处理。这是初步答复:..." |

→ **真正的"AI 私人秘书"**,24/7 待命。

---

## ⭐ 5.4.5 n8n 优势与局限性

### ✅ 优势

| 优势                  | 解释                       |
| ------------------- | ------------------------ |
| **连接能力极强**          | 数百预置节点,**几乎所有 SaaS 都能接** |
| **开源 + 可自托管**       | 数据安全                     |
| **AI Agent 节点设计精妙** | LLM + Memory + Tools 一体化 |
| **真正的"工作流"思维**      | 适合集成进现有业务                |
| **触发器丰富**           | 邮件/定时/Webhook/数据库变化 等等   |

### ⚠️ 局限

| 局限 | 解释 |
|---|---|
| **学习曲线陡** | 节点配置需要懂 JSON / 表达式 |
| **AI 能力专一度不如 Coze/Dify** | 毕竟主业是自动化 |
| **UI 比 Coze 复杂** | 不适合非技术新手 |
| **JS 表达式可能挡门** | 复杂场景需要写 `{{ $json.xxx }}` |
| **国内访问偶有不稳** | 部分服务被墙 |

---

## 🎯 5.4.6 何时用 n8n?

```
✅ 适合:
- 已经在用各种 SaaS 工具(Notion/Slack/Gmail/数据库)
- 想把 AI **嵌入现有业务流程**
- 需要复杂的触发/定时/Webhook 集成
- 对数据安全有要求(自托管)
- 有工程经验,愿意学

❌ 不适合:
- 纯做 AI 聊天机器人(用 Coze)
- 企业级 AI 应用平台需求(用 Dify)
- 完全没编程经验
- 只想要 AI 不要其他工具
```

---

## ⚠️ 小白避坑

1. **节点之间的数据格式**
   - n8n 用 JSON 在节点间传递
   - 字段名必须用 `{{ $json.xxx }}` 引用
   - **写错字段名 → 静默失败,极难调试**
2. **OAuth 配置复杂**
   - Gmail/GitHub 等服务需要 OAuth 凭证
   - 第一次配置要 30 分钟,**做好心理准备**
3. **AI Agent 节点的 Memory Key 唯一性**
   - 多个会话共用一个 Key → **对话混乱**
   - 用 `threadId` 等业务 ID 作 Key 是好实践
4. **测试时用真实 Trigger**
   - "模拟 Webhook"和真实 Webhook 行为可能不同
5. **n8n 云版有任务执行数限制**
   - 免费 5,000/月,**生产用得自托管**

---

## 📌 5.4 节要点

| 知识点 | 一句话 |
|---|---|
| **n8n 定位** | 工作流自动化 + AI 嵌入 |
| **节点 + 工作流** | 数百节点 + JSON 数据流 |
| **AI Agent 节点** | LLM + Memory + Tools 一体化 |
| **触发器** | Gmail / 定时 / Webhook 等 |
| **典型应用** | 把 AI 嵌入现有业务流程 |
| **杀手锏** | **连接万物**(几百种 SaaS) |
| **不适合** | 纯做 AI 聊天 |

---

## 🔗 延伸阅读

- 上一节:[03-Dify-开源全栈平台](03-Dify-%E5%BC%80%E6%BA%90%E5%85%A8%E6%A0%88%E5%B9%B3%E5%8F%B0.md)
- 下一节:[05-三大平台对比与选型](05-%E4%B8%89%E5%A4%A7%E5%B9%B3%E5%8F%B0%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)
- 官方:https://n8n.io
- 类似产品:Zapier / Make.com / Pipedream

---

⬅ [03-Dify-开源全栈平台](03-Dify-%E5%BC%80%E6%BA%90%E5%85%A8%E6%A0%88%E5%B9%B3%E5%8F%B0.md)	|	➡ [05-三大平台对比与选型](05-%E4%B8%89%E5%A4%A7%E5%B9%B3%E5%8F%B0%E5%AF%B9%E6%AF%94%E4%B8%8E%E9%80%89%E5%9E%8B.md)
