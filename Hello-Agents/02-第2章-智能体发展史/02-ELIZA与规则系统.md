---
tags: [Hello-Agents, 第2章, ELIZA, 规则系统, 实战]
chapter: 2
section: 2.2
---

# 2.2 构建基于规则的聊天机器人(ELIZA)

⬅ [01-符号主义早期智能体](01-%E7%AC%A6%E5%8F%B7%E4%B8%BB%E4%B9%89%E6%97%A9%E6%9C%9F%E6%99%BA%E8%83%BD%E4%BD%93.md)	|	➡ [03-心智社会](03-%E5%BF%83%E6%99%BA%E7%A4%BE%E4%BC%9A.md)

---

## 🎬 故事比喻:1966 年的"假心理医生"

1966 年,**MIT 的魏泽鲍姆**写了个程序叫 **ELIZA**,扮演**心理治疗师**。
你跟它说:"我对我妈感到生气"
它回:"**告诉我更多关于你妈的事**"

→ 你以为它在共情你?**它压根不懂你说了啥**。
→ 它只是用**模式匹配 + 句式转换**装出共情的样子。

**最讽刺的是**:许多用户(包括魏泽鲍姆自己的秘书)对 ELIZA **产生了情感依赖** —— 这就是著名的 **"ELIZA 效应"**。

---

## 🧠 2.2.1 ELIZA 的设计思想

### 核心目标

> **不是真正理解,而是用句式转换技巧,制造"理解"的假象。**

### 工作机制

```
用户说: "我为我的男朋友感到难过"
   ↓
识别关键词: "我为...感到难过"
   ↓
应用转换规则
   ↓
ELIZA 回: "你为什么会为你的男朋友感到难过?"
```

→ 表面看起来很有共情,**实际只是把陈述转成反问**。

### 设计哲学

魏泽鲍姆**故意做这个程序**,是为了**证明"机器对话能力是假象"**。
结果**啪啪打脸** —— 用户真的相信 ELIZA 懂自己。

→ 这件事**直接催生了"AI 伦理"研究**。

---

## 🔧 2.2.2 算法流程:4 步搞定

### Step 1:关键词识别 + 排序

规则库为每个关键词设**优先级**:
```
高优先级: mother, depressed, dreamed
低优先级: yes, no, hello
```
匹配时按**优先级最高**的关键词应用规则。

### Step 2:分解规则(用通配符)

```
规则: * my *
输入: "My mother is afraid of me"
匹配后捕获: ["", "mother is afraid of me"]
```

### Step 3:重组规则(生成回应)

```
规则模板: "Tell me more about your family."
直接输出: "Tell me more about your family."
```

或者**带上捕获内容**:
```
规则模板: "Why do you feel {captured}?"
输入捕获: "sad about my work"
输出: "Why do you feel sad about my work?"
```

### Step 4:代词转换(维持对话连贯)

```
I → you
my → your
am → are
```

例:
```
输入: "I am afraid of my future"
代词转换后: "you are afraid of your future"
重组: "Why are you afraid of your future?"
```

---

## 🐍 2.2.3 用 ~70 行 Python 复现 ELIZA

```python
import re
import random

# ⭐ 规则库: 模式(正则) → 响应模板列表
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    r'.*': [   # ⭐ 兜底规则:啥也没匹配上时用
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# ⭐ 代词转换表
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "i'd": "you would",
    "yours": "mine", "mine": "yours"
}

def swap_pronouns(phrase):
    """把捕获到的句子做代词转换 (I → you, my → your)"""
    words = phrase.lower().split()
    return " ".join([pronoun_swap.get(w, w) for w in words])

def respond(user_input):
    """根据规则库生成响应"""
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 捕获括号里的部分
            captured = match.group(1) if match.groups() else ''
            # 代词转换
            swapped = swap_pronouns(captured)
            # 随机选一个模板 + 填空
            return random.choice(responses).format(swapped)
    # 兜底
    return random.choice(rules[r'.*'])


# 主循环
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye.")
            break
        print(f"Therapist: {respond(user_input)}")
```

### 真实运行

```
Therapist: Hello! How can I help you today?
You: I am feeling sad today.
Therapist: How long have you been feeling sad today?
You: I need some help with my project.
Therapist: Are you sure you need some help with your project?
You: My mother is not happy with my work.
Therapist: Tell me more about your mother.
You: quit
Therapist: Goodbye.
```

→ 看起来很像心理医生,**但它完全不懂你**。

### 📝 代码解读

#### 关键点 1:规则按字典顺序匹配
Python 3.7+ 字典**保持插入顺序**,**特殊规则要放前面**。
否则兜底 `r'.*'` 会先匹配,所有输入都得到通用回复。

#### 关键点 2:`re.search` 而非 `re.match`
`search` 在**字符串任意位置**找匹配,`match` 只从开头找。
这样 "I am feeling sad" 能被 `I am (.*)` 匹配,而不需要严格开头。

#### 关键点 3:`{0}` 格式化填空
```python
"How long have you been {0}?".format("feeling sad")
# → "How long have you been feeling sad?"
```

#### 关键点 4:`random.choice` 增加多样性
同一个模式有多个响应模板,**随机选一个**让对话不那么机械。

---

## ❌ 通过实战,我们直观看到规则系统的根本局限

### 局限 1:**缺乏语义理解**

```
You: I am NOT happy
Therapist: How long have you been NOT happy?
```

→ ELIZA **不理解 "NOT"**,机械地生成无意义回应。

### 局限 2:**无上下文记忆**(Stateless)

```
You: My name is Alice.
Therapist: Tell me more about your family.

You: What's my name?
Therapist: Please tell me more.       ← 完全忘了你叫 Alice!
```

→ ELIZA **不记得上一句说了啥**。

### 局限 3:**规则爆炸**

想覆盖更多场景?加规则。
但规则一多:
- **冲突**(同一句话匹配多个规则)
- **优先级混乱**
- **维护噩梦**

→ 这就是 ["知识获取瓶颈"](01-%E7%AC%A6%E5%8F%B7%E4%B8%BB%E4%B9%89%E6%97%A9%E6%9C%9F%E6%99%BA%E8%83%BD%E4%BD%93.md#-214-%E7%AC%A6%E5%8F%B7%E4%B8%BB%E4%B9%89%E7%9A%84%E6%A0%B9%E6%9C%AC%E6%80%A7%E6%8C%91%E6%88%98) 的直接体现。

---

## 🎭 "ELIZA 效应" —— 智能的幻觉

尽管 ELIZA 这么简单,**用户却产生了真情感**。原因:

| 因素          | 解释                        |
| ----------- | ------------------------- |
| **被动提问者角色** | 心理医生本来就少说多问               |
| **开放式模板**   | "告诉我更多" 让用户自己补全意义         |
| **人类情感投射**  | 我们天生爱**把情感投射到回应自己的"东西"上** |

→ 70 年代就**警示了 AI 伦理**:**用户可能把"会说话"的程序**当成有意识的存在。

→ 今天 ChatGPT 也面临同样问题(放大 100 倍)。

---

## 🆚 ELIZA vs ChatGPT 的本质差异

| 维度 | ELIZA(1966) | **ChatGPT(2022+)** |
|---|---|---|
| 工作原理 | 模式匹配 + 模板 | 神经网络 + 注意力 |
| 理解能力 | ❌ 完全不懂语义 | ✅ 有(虽然不完美) |
| 上下文记忆 | ❌ 单句无状态 | ✅ 长上下文 |
| 知识来源 | 手写规则 | 海量预训练 |
| 应对未知 | 失灵 | 涌现能力 |
| 规则数量 | 几十~几百 | **十亿~千亿参数** |

→ 都能"对话",但**底层范式完全不同**。

---

## ⚠️ 小白避坑

1. **不要小看 ELIZA**
   - 它启发了**对话系统 + AI 伦理**整个领域
   - 它的**人格化设计**到今天还在用(ChatGPT 的"友好"语气)
2. **规则系统没死**
   - 智能客服里**FAQ 兜底**仍是规则匹配
   - 任务型对话(订餐/订机票)**槽位填充**也是规则
3. **正则的局限**
   - 复杂语言现象正则**写不完**
   - 这就是为什么神经网络最终胜出

---

## 📌 2.2 节要点

| 知识点 | 一句话 |
|---|---|
| ELIZA | 1966 年的"假心理医生" |
| 核心机制 | **模式匹配 + 代词转换 + 模板填空** |
| ELIZA 效应 | 人类天生爱投射情感到对话对象 |
| 根本局限 | 无语义 + 无记忆 + 规则爆炸 |

---

## 🔗 延伸阅读

- 上一节:[01-符号主义早期智能体](01-%E7%AC%A6%E5%8F%B7%E4%B8%BB%E4%B9%89%E6%97%A9%E6%9C%9F%E6%99%BA%E8%83%BD%E4%BD%93.md)
- 下一节:[03-心智社会](03-%E5%BF%83%E6%99%BA%E7%A4%BE%E4%BC%9A.md) —— 多智能体思想之源
- 现代对比:[第 1 章 LLM Agent 旅行助手](../01-%E7%AC%AC1%E7%AB%A0-%E5%88%9D%E8%AF%86%E6%99%BA%E8%83%BD%E4%BD%93/03-%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0%E6%97%85%E8%A1%8C%E5%8A%A9%E6%89%8B.md)

---

⬅ [01-符号主义早期智能体](01-%E7%AC%A6%E5%8F%B7%E4%B8%BB%E4%B9%89%E6%97%A9%E6%9C%9F%E6%99%BA%E8%83%BD%E4%BD%93.md)	|	➡ [03-心智社会](03-%E5%BF%83%E6%99%BA%E7%A4%BE%E4%BC%9A.md)
