---
tags: [Hello-Agents, 第4章, Reflection, 范式, 实战]
chapter: 4
section: 4.4
---
 
# 4.4 Reflection 范式(完整实现)

⬅ [03-Plan-and-Solve范式](03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md)	|	➡ [05-三大范式对比](05-%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F%E5%AF%B9%E6%AF%94.md)

> **"做完再想"** 的范式 —— 写完初稿后**自我批判 + 修订**,迭代优化。
> Noah Shinn 等 2023 年提出 **Reflexion** 框架 [3]。

---

## 🎬 故事比喻:写论文的人

| 范式 | 比喻 |
|---|---|
| **ReAct** | 侦探(边查边推理) |
| **Plan-and-Solve** | 建筑师(先画蓝图) |
| **Reflection** | **作家**(写完初稿 → 自己审 → 改 → 再审 → 再改) |

→ 写论文从来不是一稿过关。**Reflection 把这种"打磨"机制引入 Agent**。

---

## 🧠 4.4.1 核心思想:执行 → 反思 → 优化

### 灵感来源:**人类的"事后校对"**

- 写完代码会**测试 + 调试**
- 解完数学题会**验算**
- 写完邮件会**通读 + 修改措辞**

**LLM 也应该有这种能力!**

### 三步循环

```
┌──────────────────┐
│   执行 Execution  │   生成初稿
│   (ReAct/Plan/…)  │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│   反思 Reflection │   评审员视角:
│                   │   - 事实性错误?
│                   │   - 逻辑漏洞?
│                   │   - 效率问题?
│                   │   - 遗漏信息?
└─────────┬────────┘
          │ 生成反馈 Feedback
          ▼
┌──────────────────┐
│   优化 Refinement │   根据反馈修订
│                   │   生成新版本
└─────────┬────────┘
          │
          ▼ 不满意?继续循环
        (达到最大轮次 / 反思无新问题 → 退出)
```

### 形式化表达

设 y_t 是第 t 轮的输出(y_0 是初稿)。

**反思**:
```
feedback_t = LLM_critic(task, y_t)
```

**优化**:
```
y_{t+1} = LLM_refine(task, y_t, feedback_t)
```

直到达到迭代上限或反思无新问题。

---

## ⭐ 4.4.2 Reflection 的独特价值

| 价值         | 解释                      |
| ---------- | ----------------------- |
| **内部纠错回路** | 不依赖外部工具反馈,**修高层逻辑错误**   |
| **持续优化过程** | 一次性任务 → 迭代精炼,**质量大幅提升** |
| **短期记忆**   | 整个"执行-反思"轨迹**就是经验**     |
| **多模态扩展**  | 可反思**代码、图像、设计**等任何输出    |

---

## 🎯 4.4.3 实战目标 + 案例设定

任务:让 Agent 写一段代码或回答一个问题,然后**自己反思** + **优化**。

PDF 用的是"**故事创作**"案例:
> "请写一个关于宇航员发现外星生命的短篇故事"

**为什么选创作类任务?**
- **没有标准答案** → 反思空间大
- **质量主观** → 能体现 Reflection 的"打磨"价值
- **不需要工具** → 聚焦反思机制本身

---

## 🐍 4.4.4 Reflection Agent 实现

### Step 1:三个 Prompt 模板

#### 模板 1:**执行(Executor)**
```python
EXECUTION_PROMPT_TEMPLATE = """
你是一位优秀的作家。请根据以下要求完成创作任务。

任务: {task}

请直接输出你的创作内容,不要添加额外的解释。
"""
```

#### 模板 2:**反思(Critic)**
```python
REFLECTION_PROMPT_TEMPLATE = """
你是一位严谨、苛刻的资深编辑。请评审以下作品。

# 原始任务:
{task}

# 待评审作品:
{draft}

# 评审重点:
1. 内容是否完整地回应了任务要求?
2. 是否存在逻辑漏洞或事实错误?
3. 表达是否清晰、流畅?有无可改进之处?
4. 是否有遗漏的关键信息或角度?

如果作品已经非常完美,请仅输出: "无需修改"。
否则,请输出具体、可操作的修改建议:
"""
```

#### 模板 3:**优化(Refiner)**
```python
REFINEMENT_PROMPT_TEMPLATE = """
你是一位优秀的作家。请根据反馈修改以下作品。

# 原始任务:
{task}

# 上一版作品:
{draft}

# 评审反馈:
{feedback}

请认真考虑反馈,输出修改后的新版作品(直接输出内容,不要解释):
"""
```

### Step 2:核心循环

```python
class ReflectionAgent:
    def __init__(self, llm_client, max_iterations: int = 3):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.memory = []   # ⭐ 短期记忆:记录所有迭代

    def run(self, task: str):
        """执行 → 反思 → 优化 循环"""
        print(f"\n--- 开始任务 ---\n任务: {task}")
        self.memory = []

        # 1. 生成初稿
        print(f"\n--- 第 0 轮: 生成初稿 ---")
        draft = self._execute(task)
        self.memory.append({"iteration": 0, "draft": draft, "feedback": None})

        # 2. 反思 + 优化 循环
        for i in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {i} 轮: 反思 + 优化 ---")

            # 反思
            feedback = self._reflect(task, draft)
            print(f"📝 反馈: {feedback}")

            # ⭐ 早停:如果说"无需修改",停止
            if "无需修改" in feedback:
                print("✅ 评审员认为已完美,提前结束")
                self.memory.append({"iteration": i, "draft": draft, "feedback": feedback})
                break

            # 优化
            new_draft = self._refine(task, draft, feedback)
            draft = new_draft
            self.memory.append({"iteration": i, "draft": draft, "feedback": feedback})

        print(f"\n--- 任务完成 ---")
        print(f"最终作品:\n{draft}")
        return draft

    def _execute(self, task: str) -> str:
        """生成初稿"""
        prompt = EXECUTION_PROMPT_TEMPLATE.format(task=task)
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages, temperature=0.7) or ""

    def _reflect(self, task: str, draft: str) -> str:
        """反思评审"""
        prompt = REFLECTION_PROMPT_TEMPLATE.format(task=task, draft=draft)
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages, temperature=0) or ""

    def _refine(self, task: str, draft: str, feedback: str) -> str:
        """根据反馈修订"""
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            task=task, draft=draft, feedback=feedback
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm_client.think(messages=messages, temperature=0.7) or ""
```

### 📝 关键设计深度解读

#### 🔑 关键 1:**3 个 Prompt 体现 3 个不同角色**
```python
EXECUTION_PROMPT  → "你是优秀作家"      (创作者视角)
REFLECTION_PROMPT → "你是严谨编辑"      (批判者视角)
REFINEMENT_PROMPT → "你是优秀作家"      (修订者视角)
```

→ **角色切换是 Reflection 的灵魂**。
→ 同一个 LLM **演不同角色**,产生不同观点。

#### 🔑 关键 2:**Temperature 的精妙设计**

```python
def _execute(self, task):
    return self.llm_client.think(messages, temperature=0.7)   # 创作:高随机

def _reflect(self, task, draft):
    return self.llm_client.think(messages, temperature=0)     # 反思:严谨

def _refine(self, task, draft, feedback):
    return self.llm_client.think(messages, temperature=0.7)   # 修订:再创作
```

| 阶段 | Temperature | 原因 |
|---|---|---|
| 创作 | 0.7 | **有创意** |
| 反思 | 0.0 | **稳定、严谨** |
| 修订 | 0.7 | **保留创意,加入改进** |

→ 这是 Reflection 工程上的**关键细节**。

#### 🔑 关键 3:**早停机制(Early Stop)**

```python
if "无需修改" in feedback:
    break
```

→ 反思员认为已完美 → **立即停止**。
→ 防止**无限优化**(同样浪费 token)。

#### 🔑 关键 4:**Memory 短期记忆**

```python
self.memory.append({"iteration": i, "draft": draft, "feedback": feedback})
```

→ 记录每一轮的"草稿 + 反馈" → 形成**修订轨迹**。
→ 可用于:
- 调试(看哪轮改坏了)
- 分析(哪种反馈最有用)
- 训练数据(给未来的 RLHF 用)

---

## 🎮 4.4.5 运行实例

### 模拟输出(展示 3 轮迭代)

```
任务: 请写一个关于宇航员发现外星生命的短篇故事

--- 第 0 轮: 生成初稿 ---
🧠 调用 LLM...
✅ 初稿:
[宇航员小明在火星看到一只长得像章鱼的生物。他很惊讶,
跑回飞船报告了任务中心。结束。]

--- 第 1 轮: 反思 + 优化 ---
📝 反馈:
1. 故事过于简短,缺乏紧张感和细节描写
2. 角色塑造不足,只是一个名字
3. 没有内心描写,无法引起共鸣
4. 缺少环境描写,场景感弱
建议:增加内心独白、环境细节、紧张氛围

🧠 调用 LLM 修订...
✅ 第 1 版:
[小明走出舱门,火星粗糙的红色岩石在脚下嘎吱作响。
他的呼吸在头盔里回响。突然,远处岩石间一抹蓝色的光闪动……
他屏住呼吸,缓缓走近。那是一只生物,八条触手在岩石上
柔和地波动……(完整故事 300 字)]

--- 第 2 轮: 反思 + 优化 ---
📝 反馈:
1. 故事开头很好,但结尾仓促
2. 没有展示"发现"带来的情感冲击
3. 缺少与生物互动的细节
建议:扩展结尾,描写情感震撼

🧠 调用 LLM 修订...
✅ 第 2 版:
[在第 1 版基础上扩展了 200 字结尾,加入了小明的情感震撼
和生物的反应描写]

--- 第 3 轮: 反思 + 优化 ---
📝 反馈: 无需修改
✅ 评审员认为已完美,提前结束

--- 任务完成 ---
最终作品: [第 2 版的完整故事]
```

→ **从平淡初稿 → 高质量作品**,经过 2 轮迭代显著提升。

---

## ⭐ 4.4.6 Reflection 的成本收益分析

### 成本

| 项            | 增加倍数                         |
| ------------ | ---------------------------- |
| **LLM 调用次数** | **3× 起**(初稿 + 反思 + 修订;每轮 ×3) |
| **Token 消耗** | 反思 prompt 含完整草稿,**Token 飞涨** |
| **延迟**       | 串行多轮,**用户等更久**               |

### 收益

| 任务类型 | 质量提升 |
|---|---|
| **创作类**(写作 / 故事) | ⭐⭐⭐⭐⭐ |
| **代码生成** | ⭐⭐⭐⭐(找出 bug) |
| **复杂推理** | ⭐⭐⭐⭐ |
| **简单事实问答** | ⭐(几乎没用) |
| **工具调用** | ⭐⭐(改改参数) |

### 何时该用?

```
任务对质量要求高?           ─是→ 用 Reflection
任务用户能接受多等几秒?      ─是→ 用 Reflection
任务复杂,容易出隐性错误?     ─是→ 用 Reflection
简单查询?                  ─是→ 不用!浪费钱
```

---

## 🔀 4.4.7 Reflection 与其他范式结合

Reflection **不是替代** ReAct / Plan-and-Solve,而是**叠加**:

| 组合 | 效果 |
|---|---|
| **ReAct + Reflection** | 工具调用后反思结果是否正确 |
| **Plan-and-Solve + Reflection** | 执行完成后反思整个解答 |
| **多 Agent + Reflection** | 一个 Agent 干,另一个 Agent 评 |

→ 真实生产系统**大多用组合**。
→ Reflexion 论文里就是 ReAct + Reflection。

---

## ⚠️ 小白避坑

1. **不要无限反思**
   - `max_iterations = 2~3` 通常够
   - 过多反思**收益递减 + 成本爆炸**
2. **反思 prompt 要明确评审维度**
   - 模糊的"找问题"会让 LLM 找不到具体毛病
   - 列出**具体检查项**效果好
3. **早停机制很重要**
   - 没有早停 → 永远到 max_iterations
   - 检测"无需修改"等关键词
4. **反思可能越改越差!**
   - LLM 有时会"过度修订"破坏好的部分
   - 解法:**保留 N 个版本,最后选最优**
5. **创作类任务别用 temperature=0**
   - 反思可以用 0(严谨),创作要 0.7+(有创意)

---

## 📌 4.4 节要点

| 知识点 | 一句话 |
|---|---|
| **三步循环** | 执行 → 反思 → 优化 |
| **3 个 prompt** | Executor / Critic / Refiner(同 LLM 不同角色) |
| **Temperature 差异化** | 创作高,反思低 |
| **早停机制** | "无需修改" → 退出 |
| **Memory 短期记忆** | 记录所有迭代,可分析改进 |
| **适合场景** | 创作 / 代码 / 复杂推理 |
| **不适合** | 简单问答 |
| **常组合使用** | ReAct/Plan + Reflection |

---

## 🔗 延伸阅读

- 上一节:[03-Plan-and-Solve范式](03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md)
- 下一节:[05-三大范式对比](05-%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F%E5%AF%B9%E6%AF%94.md) —— 总结 + 选型
- 论文:Reflexion (Shinn et al., 2023)

---

⬅ [03-Plan-and-Solve范式](03-Plan-and-Solve%E8%8C%83%E5%BC%8F.md)	|	➡ [05-三大范式对比](05-%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F%E5%AF%B9%E6%AF%94.md)
