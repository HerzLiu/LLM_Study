---
tags: [Code-Agent, 原始资料, Coder-Model, 训练全流程]
status: Archived
source: "微信临时文件：训练Coder模型全流程.md"
stage: appendix
---

# 从零训练一个 Coder Model：数据 / 预训练 / Mid-train / SFT / RL 全流程

> 假设你要从一个通用 base 模型出发，训练一个能做 Agentic Coding（自主修 bug、写代码、跑测试）的 coder model。本文按 **数据形态 → 预训练 → Mid-train → SFT → RL** 的真实链路，逐段讲清每个阶段「喂什么数据、怎么训、有哪些坑」。

---

## 目录

- [〇、全局视角：四阶段在干嘛](#〇全局视角四阶段在干嘛)
- [一、数据：每个阶段长什么样、有哪些类](#一数据每个阶段长什么样有哪些类)
- [二、预训练 Pre-train](#二预训练-pre-train)
- [三、Mid-train（continued pre-train / 退火）](#三mid-traincontinued-pre-train--退火)
- [四、SFT 训练过程](#四sft-训练过程)
- [五、RL 训练过程（完整环境 + 训练细节）](#五rl-训练过程完整环境--训练细节)
- [六、一张图串起全流程](#六一张图串起全流程)

---

## 〇、全局视角：四阶段在干嘛

```
Base模型 ──→ ① Pre-train ──→ ② Mid-train ──→ ③ SFT ──→ ④ RL ──→ Coder Model
            学代码知识      注入能力/退火    学会按格式干活  自我提升修对率
            (海量代码)     (高质量+工具)    (带答案轨迹)   (测试当reward)
```

| 阶段 | 一句话目标 | 数据信号 | 学习率量级 | 数据量级 |
|---|---|---|---|---|
| ① Pre-train | 让模型「懂代码」 | next-token 预测 | 1e-4 ~ 3e-4 | 1000B+ tokens |
| ② Mid-train | 注入工具调用 / 长上下文 / 退火到高质量 | next-token（高质量子集） | 3e-4 → 衰减 | 50B~200B tokens |
| ③ SFT | 学会「按 agent 格式多轮干活」 | 模仿教师轨迹（cross-entropy） | 1e-5 ~ 2e-5 | 50K~500万条 |
| ④ RL | 自己探索把 bug 真修对 | 测试 pass/fail → reward | 1e-6 ~ 5e-6 | 1万~5万 task |

核心直觉：**Pre-train/Mid-train 灌知识，SFT 教格式与行为，RL 真正提升「解决问题的成功率」**。前三个是「模仿」，只有 RL 是「试错 + 反馈」。

---

## 一、数据：每个阶段长什么样、有哪些类

### 1. Pre-train 数据：纯文本 next-token

**形态**：无标注的原始代码 / 文本，模型做 next-token 预测，没有「问答」结构。

**有哪些类**：

| 数据类 | 内容 | 作用 |
|---|---|---|
| **原始代码** | GitHub 海量仓库源码（多语言） | 学语法、API、常见模式 |
| **Repo-level 代码** | 同一仓库多文件按依赖顺序拼接 | 学跨文件理解（import 依赖、调用链） |
| **代码相关文本** | 文档、README、StackOverflow、教程 | 学「代码 + 自然语言」对齐 |
| **通用网页文本** | 高质量 Web 语料 | 防止只会代码、丢失通用语言能力 |
| **数学 / 推理** | 数学题、逻辑推理 | 代码能力和推理能力强相关 |

**各类数据示例**：

**① 原始代码**（一个文件原样喂入，模型预测下一个 token）：

```python
# 文件: numpy/lib/function_base.py（整段源码直接当训练文本）
def average(a, axis=None, weights=None, returned=False):
    a = np.asanyarray(a)
    if weights is None:
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = np.asanyarray(weights)
        ...
```

**② Repo-level 代码**（同仓库多文件按 import 依赖拓扑序拼接成一条长样本）：

```text
<|repo_name|>myproject
<|file_sep|>utils/math_helpers.py        ← 被依赖的文件排前面
def add(a, b):
    return a + b
<|file_sep|>core/calculator.py            ← 依赖 utils 的文件排后面
from utils.math_helpers import add
class Calculator:
    def run(self, x, y):
        return add(x, y)
<|file_sep|>main.py
from core.calculator import Calculator
Calculator().run(1, 2)
```

**③ 代码相关文本**（文档 / StackOverflow 问答，教「代码 ↔ 自然语言」对齐）：

```text
## How do I merge two dicts in Python 3.9+?
You can use the union operator `|`:

    d1 = {"a": 1}
    d2 = {"b": 2}
    merged = d1 | d2   # {"a": 1, "b": 2}

Before 3.9 you had to use `{**d1, **d2}` or `dict.update()`.
```

**④ 通用网页文本**（普通高质量自然语言，防止丢失通用语言能力，无代码）：

```text
气候变化是指地球气候系统长期的统计学变化。工业革命以来，
人类活动排放的温室气体是当前全球变暖的主要驱动因素……
```

**⑤ 数学 / 推理**（数学推导，和代码推理能力强相关）：

```text
求 ∫ x·e^x dx。用分部积分，令 u=x, dv=e^x dx，
则 du=dx, v=e^x，原式 = x·e^x − ∫ e^x dx = x·e^x − e^x + C。
```

**【重点】两个 code 专属的训练目标**（不只是从左到右预测）：

```text
① FIM（Fill-in-the-Middle）填空
   把代码切成 prefix / middle / suffix，让模型根据前后文补中间
   形如：<PRE> def add(a,b): <SUF> return result <MID> result = a+b
   → 这是代码补全（IDE 里的自动补全）能力的来源

② Repo-level 拓扑序拼接
   解析 import 依赖建 DAG → 拓扑排序，让「被依赖的文件」排在前面
   用 <|file_sep|> 分隔多个文件拼成一条长样本
   → 让模型学到「先有 utils.py 才有 main.py 调它」的真实结构
```

> **它们在「数据」和「Loss」上特殊在哪？** 一句话：**loss 函数和普通预训练完全一样（都是标准 next-token 交叉熵），特殊性全在「数据怎么排列」——改的是数据，不是 loss 公式**。

**FIM 的特殊点**：

- **数据上（做了一次序列重排）**：把文件切成 `prefix / middle / suffix`，再**把 middle 挪到序列末尾**重新拼接。这样模型自回归预测 middle 时，上文里同时有了 prefix 和 suffix，等于学会「看前后文填空」。标准做法叫 PSM（Prefix-Suffix-Middle）+ SPM（Suffix-Prefix-Middle）各一半。
  ```text
  原始:  def add(a,b):\n  result = a+b\n  return result
  重排:  <PRE> def add(a,b):\n  <SUF> \n  return result <MID> result = a+b
         └─前缀──────────┘     └─后缀────────────┘      └─中间(要预测)┘
  ```
- **Loss 上（公式不变）**：还是逐 token 交叉熵；唯一讲究是分隔符 token（`<PRE>/<SUF>/<MID>/<EOT>`）也算 loss，让模型学会「该停的地方停」；有些实现会把 prefix/suffix 段 loss 权重调低、重点压在 middle。
- **为什么**：普通从左到右训练只会续写、不会回填，而 IDE 补全是「光标在中间、前后都有代码」。FIM 纯靠数据重排获得回填能力，不动结构和 loss。

**Repo-level 拓扑序的特殊点**：

- **数据上（跨文件拼接 + 依赖排序）**：普通预训练「一个文件一条独立样本」；Repo-level 把同仓库多文件拼成一条长样本，用 `<|file_sep|>` 分隔，并**按 import 依赖拓扑排序**——被依赖的文件排前面。
  ```text
  <|file_sep|>utils.py        ← 先出现（被依赖）
  def add(a, b): return a + b
  <|file_sep|>main.py          ← 后出现（依赖 utils）
  from utils import add        ← 预测 add 时，上文真的见过它的定义
  add(1, 2)
  ```
- **Loss 上（公式不变，但跨文件引用变成有效监督）**：还是 next-token 交叉熵；因为被依赖文件排前面，模型预测 `main.py` 里的 `add` 时上文有定义，这个 token 的 loss 等价于在**监督「跨文件符号引用」**——孤立文件训练根本学不到。反例：随机顺序拼接（main 在 utils 前），预测 add 时上文无定义，信号就成了噪声。**拓扑排序就是为了让这个 loss 信号有意义**（循环依赖时选 import 最少者前置）。

| | 数据上的特殊 | Loss 上的特殊 |
|---|---|---|
| **FIM** | 切 prefix/suffix/middle，**把 middle 重排到末尾**（PSM/SPM 各半） | 公式不变（CE）；分隔符 token 也算 loss，学会何时停 |
| **Repo-level** | **多文件拼接 + import 拓扑排序**，被依赖文件排前 | 公式不变（CE）；跨文件引用因上文有定义而成有效监督信号 |

> **核心洞察**：这两个都是**「数据工程」而非「loss 工程」**——loss 始终是标准 next-token 交叉熵，特殊性全部来自**对训练序列的重新排列**，让模型自回归预测时「恰好用上」原本学不到的信息（前后文填空、跨文件依赖）。这也是它们能无缝混进普通预训练、不需改训练框架的原因。

### 2. Mid-train 数据：高质量子集 + 工具轨迹

形态还是 next-token，但**数据质量和配比变了**，并开始混入「带工具调用格式的轨迹」：

| 数据类 | 内容 | 为什么 mid-train 放 |
|---|---|---|
| **高质量代码** | 经过质量过滤的精选代码（top 质量分） | 退火阶段拉高下限 |
| **工具调用轨迹** | 规范化的 `tool_call` 格式样本（5-10%） | 让 base 提前学会工具调用的 token 分布 |
| **单格式 CoT** | 统一 `<think>...</think>` 思考格式 | 给后续 reasoning 能力打底 |
| **长上下文样本** | 长文件 / 长 repo 拼接（4K→200K 渐增） | 训长上下文外推 |

> 关键经验：**工具调用要在 Mid-train（LR 3e-4 塑性强）注入，而不是 SFT（LR 2e-5 只学表面）**。Mid-train 注入后 base 模型 tool 字段合规率能从 ~70% 提到 ~95%，下游 SFT 数据量直接减半。

**各类数据示例**：

**① 工具调用轨迹**（注意：mid-train 阶段仍是 next-token，工具调用是**当成纯文本序列**喂进去，让模型熟悉 `<tool_call>` 这类特殊 token 的分布；还不强调多轮对话结构）：

```text
用户想查询北京天气并计算温差。
<tool_call>
{"name": "get_weather", "arguments": {"city": "北京", "date": "today"}}
</tool_call>
<tool_response>
{"high": 28, "low": 16, "condition": "晴"}
</tool_response>
今天北京最高 28°C，最低 16°C，温差为 28 − 16 = 12°C。
```

> 重点是让模型把 `<tool_call>...</tool_call>` 的 JSON 结构、参数命名、闭合标签学成「自然的 token 序列」，覆盖多种 scaffold 的格式（OpenAI function-call 风格、XML 风格等）。

**② 单格式 CoT**（统一用一种 `<think>...</think>` 思考格式，给后续 reasoning 打底；关键是**只用一种格式**，避免格式多样性引起冲突）：

```text
问题：一个班 40 人，男生比女生多 8 人，男生几人？
<think>
设女生 x 人，则男生 x+8 人。x + (x+8) = 40 → 2x = 32 → x = 16。
所以男生 = 16 + 8 = 24 人。
</think>
男生有 24 人。
```

> **为什么强制单格式？** CoT 数据若混用多种思考标记（`<think>` / `## 推理过程` / 「让我一步步想」/ special token 等），模型在同一条 next-token 序列上被迫同时学两件事：**① 推理逻辑本身**（我们要的）+ **② 在当前上下文该用哪种格式包裹思考**（纯噪声）。后者带来三个可量化的负面效应——(a) **梯度冲突**：语义近乎相同的输入要预测不同的格式 token，梯度方向相互抵消，有效信号被稀释；(b) **解码期格式漂移**：推理时容易出现「`<think>` 开头却以 `## 答案` 收尾」的串味，导致下游正则/解析器抽取失败；(c) **容量挤占**：格式选择占用本就有限的参数容量，小模型表现尤为敏感。统一为单一格式后，格式分支熵归零，模型容量集中到推理逻辑，垂域 SFT 实测 **+2.88pp**。

> **延伸（同源但更隐蔽的坑）：reasoning / non-reasoning 必须分流，不可混训。** 除了「思考用什么标签」，还有「这条样本到底带不带思考」这一维：reasoning 样本形如 `<think>…</think> 答案`、non-reasoning 样本形如 `答案`（无思考段）。两者混训会让模型对「是否进入思考态」这一隐变量产生**双峰分布**：① non-reasoning 模型若见过 `<think>`，推理时可能进入思考态却学不到正确的终止边界，**无限输出 `<think>` 内容停不下来**；② reasoning 模型若混入大量「无思考直接答」样本，会把「跳过思考」当成高频模式，**抑制思维链的触发**，思考能力退化。工程上按 `reasoning_content` 是否为空把数据**切成两条独立 pipeline** 分别训练（甚至分别产出两套权重），从数据层面消除这个隐变量的歧义。

**③ 长上下文样本**（把整个 repo / 超长文件拼成 4K→200K 的长序列，训练窗口外推；样本内可放「跨越很远距离才能关联」的信息，逼模型用上长程依赖）：

```text
<|file_sep|>config/settings.py          ← 第 100 token 处定义
DATABASE_TIMEOUT = 30

...（中间隔着几万 token 的其他文件）...

<|file_sep|>db/connection.py            ← 第 80000 token 处使用
from config.settings import DATABASE_TIMEOUT
def connect():
    # 模型要「记得」几万 token 前定义的 DATABASE_TIMEOUT=30
    sock.settimeout(DATABASE_TIMEOUT)
```

> **长上下文是怎么「拉」上去的（长度课程 + 退火详解）**

长上下文不是一步到位，而是**分阶段渐进**，原因是窗口一旦突然放大，RoPE 位置编码外推到没训过的远距离位置，attention 分数分布骤变，loss 会出现剧烈尖峰（**loss spike**），严重时训练直接发散。所以分两条线同时渐进：

**线 A：序列长度课程（窗口逐档放大，每档配套调 RoPE base）**

| 阶段 | 训练窗口 | RoPE base（θ） | 这一档主要练什么 | 切换节点判据 |
|---|---|---|---|---|
| 基础 | 4K | 10K（原生） | 巩固原窗口能力 | 起点 |
| 扩展 1 | 32K | ~100K | 中等距离依赖 | 上一档 loss 平稳、长样本指标不再涨 |
| 扩展 2 | 128K | ~500K | 长程检索/引用 | 同上 |
| 目标 | 200K | ~500K+ | 全仓库级依赖 | 同上 |

- **什么节点调整**：不是按固定 step 切，而是**等当前档位收敛**（训练 loss 平稳 + 该长度的验证指标如 RULER 不再提升）才进下一档。每升一档**同步放大 RoPE base θ**——窗口和 θ 必须配对调整，只放大窗口不调 θ，远端位置编码会「绕圈混叠」失效。
- 每次切换通常**重新 warmup 一小段学习率**（几十~几百 step）再回到正常 LR，吸收窗口突变带来的冲击。

**线 B：长样本配比退火（5% → 25% 渐增）**

「退火（annealing）」本义是「逐渐降温让系统平稳过渡到新状态」，这里指**让某个训练参数沿训练进度平滑地从起点滑向终点，而不是一步跳变**。具体到长样本配比：

```
训练进度  ──────────────────────────────────►
长样本占比  5% ───── 8% ───── 15% ───── 25%
            │                            │
            初期主体仍是短样本           后期长样本占比拉满
            (维持原能力,不被长样本带崩)  (重点强化长程依赖)
```

- **为什么渐增而非一开始就 25%**：长样本梯度大、数值不稳，开局就高占比会冲垮已学好的短窗口能力（灾难性遗忘）+ 触发 loss spike。低占比起步让模型「先适应一点点长样本」，再缓慢加量，全程平滑过渡。
- **线 A 和线 B 的关系**：两者叠加——窗口放大到 128K 这一档内部，长样本配比也在从低往高退火。等于「窗口阶梯式跳 + 配比连续滑」双重渐进。
- **判断退火到位**：长上下文评测（RULER 128K、多文档 NIAH）达标且短任务（通用 benchmark）几乎不退化，说明长短能力平衡，退火曲线合理。

> 一句话：**长度课程**管「窗口开多大、RoPE θ 配多少、何时进下一档（看收敛）」；**退火**管「长样本占比怎么从低到高平滑爬升」。两者都是为了避免窗口/分布突变引发的 loss spike 与短能力遗忘。

### 3. SFT 数据：带答案的多轮 agent 轨迹（核心结构）

**形态**：完整的「任务 → 模型多轮思考 + 调工具 + 看结果 → 解决」对话，每一步都有标准答案，模型做模仿学习。

```json
{
  "messages": [
    {"role": "system", "content": "你是 coding agent，可用工具 bash / str_replace_editor / submit..."},
    {"role": "user", "content": "修复 requests 库 timeout=0 永久阻塞的 bug"},
    {"role": "assistant",
     "reasoning_content": "<think>先定位超时相关代码在 adapters.py</think>",
     "content": "我先搜索 timeout 的处理逻辑",
     "tool_calls": [{"name": "bash", "args": "grep -rn timeout src/"}]},
    {"role": "tool", "content": "src/requests/adapters.py:45: timeout=None",
     "content_mask": true},          // ← 工具返回，不参与 loss
    {"role": "assistant",
     "tool_calls": [{"name": "str_replace_editor",
                     "args": "adapters.py:45 改超时判断"}]},
    {"role": "tool", "content": "编辑成功", "content_mask": true},
    {"role": "assistant",
     "tool_calls": [{"name": "bash", "args": "pytest tests/test_timeout.py"}]},
    {"role": "tool", "content": "1 passed", "content_mask": true},
    {"role": "assistant", "tool_calls": [{"name": "submit"}]}
  ],
  "channel": "swe_bench",        // 任务类型标签，用于差异化 loss 权重
  "reasoning_content": "非null"  // 决定走 reasoning / standard pipeline
}
```

**SFT 数据有哪些类**（按能力域分，分开构造、分开训）：

| 域 | 任务样子 | 验证方式 |
|---|---|---|
| **SWE（修 bug）** | 给 issue 修真实仓库 | 跑测试（F2P/P2P） |
| **代码理解** | 解释代码 / 定位实现 / 追调用链 | 答案唯一性 |
| **Terminal（终端操作）** | 多步 shell 命令完成任务 | 检查文件/进程/输出状态 |
| **WebCoding（前端）** | 写网页 / UI | 渲染 + 美学评分 |
| **代码补全 / 生成** | 写函数 / 算法题 | 单元测试 |

**各域示例**（统一用「题面 → 期望轨迹要点 → 验证」缩略格式；SWE 域的完整多轮 JSON 见本节开头那段）：

**① SWE（修真实仓库 bug，跑测试 F2P/P2P）**

```text
user : [psf/requests issue] timeout=0 时请求永久阻塞，应立即超时。
       仓库 @ commit a1b2c3d，修复后需让 test_zero_timeout 通过。
assistant: <think>超时判断在 adapters.py，timeout=0 被当成 None 了</think>
   grep -rn timeout src/requests/      →  定位 adapters.py:45
   str_replace_editor 改 timeout 判断（区分 0 和 None）
   pytest tests/test_timeout.py        →  1 passed
   submit
验证：FAIL_TO_PASS（test_zero_timeout）由失败转通过，
      且 PASS_TO_PASS（test_normal_timeout）仍通过（无回归）才算解决
```

**② 代码理解**（要求精确定位 / 追调用链，答案唯一可判）

```text
user : 在 Django 仓库里，QuerySet.filter() 最终是在哪个文件、哪个方法里
       把条件编译成 SQL 的？给出文件路径 + 方法名。
assistant: <think>filter 返回新 QuerySet，真正编译 SQL 在 compiler...</think>
   grep -rn "def as_sql" django/db/models/sql/    →  定位
   读 compiler.py 的 SQLCompiler.as_sql
   答：django/db/models/sql/compiler.py :: SQLCompiler.as_sql
验证：标准答案是固定的 文件路径+方法名，字符串/路径匹配即通过
```

**③ Terminal（多步 shell，验证最终系统状态）**

```text
user : 把当前目录下所有 .log 文件里含 "ERROR" 的行，汇总去重后写到 errors.txt，
       并打印总行数。
assistant:
   grep -h "ERROR" *.log | sort -u > errors.txt
   wc -l < errors.txt
验证：检查 errors.txt 是否存在、内容是否等于「所有 ERROR 行去重」、
      stdout 末尾打印的行数是否与文件行数一致（不看过程，只看终态）
```

**④ WebCoding（前端，渲染 + 规则/美学打分）**

```text
user : 用纯 HTML+CSS 写一个登录卡片：居中、圆角阴影、含邮箱/密码输入框
       和一个蓝色登录按钮，响应式。
assistant: 输出完整 index.html（含 <style>）
验证：① 无头浏览器渲染截图，规则检查（元素是否存在、是否居中、按钮颜色）
      ② 可选 VLM/美学模型给观感打分；两者加权
```

**⑤ 代码补全 / 生成（单元测试是金标准）**

```text
user : 实现 def two_sum(nums, target) -> list[int]，返回两数之和等于 target
       的下标；保证恰有一解。
assistant: <think>哈希表存值→下标，一次遍历</think>
   def two_sum(nums, target):
       seen = {}
       for i, x in enumerate(nums):
           if target - x in seen: return [seen[target-x], i]
           seen[x] = i
验证：跑隐藏单元测试集（含边界：负数、重复值、大数组），全过才保留
```

> 五个域**共用同一套「沙箱跑 + 验证器过滤」机制**，区别只在**验证器形态**：SWE/补全用「跑测试」，代码理解用「答案精确匹配」，Terminal 用「检查终态」，WebCoding 用「渲染 + 打分」。所以扩一个新能力域 ≈ 写一个新验证器。

> **SFT 数据从哪来？** 通常是：拿一批**有验证标准的任务**（比如带测试的仓库实例），让一个**强教师模型**在沙箱里真跑一遍、跑出成功轨迹，再把成功轨迹当监督数据。**关键是只保留「真的解决了任务」的轨迹**（测试通过的），失败轨迹丢弃。这套流程也叫 **拒绝采样（rejection sampling）/ 蒸馏**。

**示例：成功轨迹的「采集 → 过滤」全流程**（下面以一条 SWE 任务示范，其余四域只是把「跑测试」换成各自验证器，流程完全一样）

```text
① 任务池（带验证标准）
   instance_id : requests__requests-1142
   repo        : psf/requests @ commit a1b2c3d
   problem     : "timeout=0 时请求永久阻塞，应立即超时"
   FAIL_TO_PASS: ["tests/test_timeout.py::test_zero_timeout"]   # 修好后必须由失败转通过
   PASS_TO_PASS: ["tests/test_timeout.py::test_normal_timeout"] # 不许改坏的回归用例

② 教师模型在沙箱里真跑（同一题采样 N 条，比如 N=8）
   ┌─ 轨迹#1: 改 adapters.py → pytest → FAIL_TO_PASS 没过      ✗ 丢弃
   ├─ 轨迹#2: 改对了 timeout 判断 → pytest → 全绿              ✓ 保留
   ├─ 轨迹#3: 改了但碰坏 PASS_TO_PASS（引入回归）              ✗ 丢弃
   ├─ 轨迹#4: 死循环搜索，超步数上限未提交                    ✗ 丢弃
   └─ ...（其余同理）                                          通常只剩 1~3 条

③ 只把「② 里测试全绿」的成功轨迹（如轨迹#2）整理成上面那种
   messages 多轮 JSON，tool 返回打 content_mask，作为 SFT 监督数据
```

- **验证器是硬门槛**：是否保留**不看教师模型「自我感觉」，只看沙箱里 F2P 转通过 + P2P 不退化**。这保证 SFT 学到的全是「真能解决问题」的轨迹，不是看着对其实跑不通的幻觉。
- **同题多采**：一题采 N 条能提高出成功轨迹的概率，也能挑「最短/最干净」的那条做监督（少走弯路）。
- **和 RL 的区别**：这里教师模型**离线**先把成功轨迹存成静态数据集再训；RL 阶段是被训模型**在线**实时 rollout、当场判分更新（见下一节）。

### 4. RL 数据：只有题面 + 验证器（没有答案）

**形态**：和 SFT 完全相反——**没有任何 assistant 轨迹**，只给「任务描述 + 怎么判分 + 在哪跑」。轨迹是模型训练时自己实时 rollout 出来的。

```json
{
  "problem_statement": "修复 requests 库 timeout=0 永久阻塞的 bug",
  "repo": "psf/requests",
  "base_commit": "abc123",
  "FAIL_TO_PASS": ["test_timeout_zero"],     // ← 修对了这些测试要从fail变pass
  "PASS_TO_PASS": ["test_normal_request"],   // ← 这些必须保持pass（防改坏）
  "docker_image": "requests-env:latest"       // ← 在这个沙箱镜像里跑
}
```

**SFT vs RL 数据格式对比**：

| | SFT 数据 | RL 数据 |
|---|---|---|
| 含答案轨迹 | ✅ 含完整教师轨迹 | ❌ 不含，模型自己生成 |
| 监督信号 | 每个 token 的 cross-entropy | 沙箱测试 pass/fail → reward |
| 核心字段 | `messages` + reasoning + channel | `problem_statement` + 测试 + docker |
| 工具返回 | masked，不算 loss | rollout 时环境的真实反馈 |
| 数据怎么造 | 教师跑成功轨迹 | 只要任务 + 可执行的判分测试 |
| 量级 | 几十万~几百万条 | 1万~5万 task（精挑有区分度的） |

---

## 二、预训练 Pre-train

**目标**：把通用 base 变成「懂代码」的 code base。

**做法要点**：

1. **多语言配比**：不是按 GitHub star 数线性分，而是考虑「每种语言的可学习难度」。算力宽裕时优先扩收益最大的语言（缩放指数大的，比如 Python）。
2. **三目标联合训练**：
   - `CLM`（标准从左到右 next-token）
   - `FIM`（填空，给 IDE 补全能力）
   - `Repo-level 拓扑序`（跨文件理解）
3. **去污染**：用 13-gram 等做去重，**确保评测集（如 SWE-bench、HumanEval）的内容没泄漏进训练数据**，否则刷分虚高。

**产物**：一个会写代码、会补全、懂仓库结构，但**还不会「按 agent 格式多轮调工具」**的 code base 模型。

---

## 三、Mid-train（continued pre-train / 退火）

夹在 Pre-train 和 SFT 之间的独立阶段，干三件事：

### 1. 退火到高质量数据
学习率按 **预热-稳定-衰减（WSD）** 调度，在 Decay 阶段切换到高质量数据子集，把模型能力下限拉高。

### 2. 注入工具调用能力
混入 5-10% 规范化的 `tool_call` 轨迹（覆盖多种 scaffold 格式）。**为什么放这里**：Mid-train 学习率 3e-4 量级塑性强，模型能真正学到 tool_use 的分布；放到 SFT（LR 2e-5）只能学到格式表面。

### 3. 长上下文外推
代码任务经常要读整个仓库，上下文窗口要从 4K 扩到 128K~200K：

```text
RoPE base 放大（10K → 500K）+ YaRN NTK-by-parts 分频段拉伸
短→长课程：4K → 32K → 128K → 200K 分阶段
长样本配比 5% → 25% 渐增退火（避免单步切换 loss spike）
评测用 RULER（多任务）而非单一 NIAH（易刷假大窗口）
```

**产物**：会代码、懂工具格式、有长窗口的增强 base，下游 SFT 更省数据。

---

## 四、SFT 训练过程

**目标**：让模型学会「按 agent 协议，多轮思考 + 调工具 + 看反馈 + 继续」的完整行为模式。

### 训练流程

```
① 准备数据：教师在沙箱跑出成功轨迹（只留测试通过的）
② 按域分治：SWE / 代码理解 / Terminal / 前端 / 补全 分别构造
③ 质量过滤（计算成本递增排序，先便宜后贵）：
     规则过滤（格式/重复/截断） → 去重 → LLM Judge 多维打分 → 保留高分
④ 训练：标准 causal LM 目标，模仿教师轨迹
```

### 四个 SFT 必须注意的训练细节

| 细节 | 做法 | 为什么 |
|---|---|---|
| **工具返回 Masking** | `tool` 消息 `content_mask=true`，不算 loss | 不能让模型去「背 / 预测 shell 输出」，只学「该调什么工具」 |
| **Reasoning/Standard 分流** | 按是否有 `<think>` 分两个独立 pipeline，**绝不混训** | 混训会让 non-reasoning 模型见 `<think>` 无限输出思考，reasoning 模型见无 thinking 样本抑制思考 |
| **Document-level Packing** | 长短不一的轨迹拼到 max_seq_len，attention 按样本边界隔离（样本间互不可见） | token 利用率 40%→95%，吞吐 1.5-2× |
| **Channel-aware Loss** | 每条打 channel 标签，差异化 loss 权重（长 SWE=1.5，短 QA=0.5） | 防止长 SWE 任务被海量短 QA 梯度淹没 |

### SFT 的局限（为什么还要 RL）

SFT 是**纯模仿**——模型学的是「教师在教师自己的轨迹上怎么走」。但推理时模型走的是**自己的轨迹**，一旦偏离教师分布就会累积误差（exposure bias）。而且 SFT 见过的成功轨迹有限，**碰到没见过的 bug 就不会修**。要真正提升「解决问题成功率」，必须上 RL 让模型自己试错。

---

## 五、RL 训练过程（完整环境 + 训练细节）

**目标**：让模型在真实可验证环境里**自己探索**，把 bug 真正修对，用「测试通过与否」当奖励信号自我提升。

### 5.1 完整环境组成（启动前要准备的东西）

```
┌──────────────────────────────────────────────────────────┐
│ ① 训练侧（verl / 类似 RL 框架）                              │
│    ├─ Actor/Policy   待训模型，FSDP分片，持梯度  ← 要更新   │
│    ├─ Rollout引擎    vLLM，高速推理生成轨迹       ← 只推理   │
│    └─ Reference模型  冻结的初始策略，算KL用        ← 不更新   │
│                                                            │
│ ② 环境侧                                                    │
│    ├─ Sandbox 池     数万级并发 Docker 沙箱（隔离执行）      │
│    ├─ Scaffold       agent 框架（SWE-agent / Claude Code）  │
│    └─ 代理层 Proxy    拦截 scaffold 的 LLM 调用转发到 vLLM   │
│                                                            │
│ ③ 数据侧                                                    │
│    └─ task 集        题面 + FAIL_TO_PASS + docker_image      │
└──────────────────────────────────────────────────────────┘
```

**关键机制：代理层劫持（保证 on-policy 的核心 trick）**

scaffold（如 SWE-agent）本来是调外部 LLM API 的。RL 训练时，把它的 API 地址指向自己的代理层：

```bash
# 让 scaffold 以为在调正常 API，实际转发到「正在训练的 vLLM」
OPENAI_BASE_URL=http://rl-proxy:8080       # SWE-agent (OpenAI 协议)
ANTHROPIC_BASE_URL=http://rl-proxy:8080    # Claude Code (Anthropic 协议)
```

这样 scaffold 全程当黑盒，它产出的每一步动作都来自待训模型，保证采样是 on-policy 的；代理层负责**协议翻译**（把各家 scaffold 格式 ↔ vLLM 格式互转）。

### 5.2 启动阶段

```
① 启动 Ray 集群，划分 Actor / Rollout / Reference 角色
② 加载 SFT 后的 checkpoint 做初始化（RL 不从零开始）
③ 起沙箱池 + scaffold + 代理层
④ 加载 task 集（只有题面+测试，没有答案）
⑤ 预筛 task：丢掉「全对 / 全错」的，只留 30-70% pass 率的有区分度样本
```

### 5.3 一个训练 Step 的完整循环（6 阶段）

以「修复 timeout=0 阻塞」为例：

**阶段 1：取 batch + 组内采样**

```
取一批 prompt（如 256 个 task）
每个 prompt 采样 G 条轨迹（GRPO group，如 G=8）
→ 256 × 8 = 2048 个并发 rollout，分发到 2048 个沙箱
```

**阶段 2：Rollout（多轮 agent 循环，最复杂）**

和普通 RL 最大不同——**一条轨迹是多轮交互**，不是一次生成：

```
沙箱里 scaffold 开始跑：
  Turn 1: 调LLM →【代理拦截→vLLM】→ "grep timeout" → 沙箱执行 → 观察
  Turn 2: 带观察再调LLM → "编辑 adapters.py:45" → 沙箱执行
  ...
  Turn N: 调 submit → 产出最终 patch
渐进 Horizon 课程：turn 上限 5→15→30→64→128 逐档放开
```

**阶段 3：Reward 计算（跑测试打分）**

```
patch apply 到 base_commit
→ 跑 FAIL_TO_PASS → 全过？bug 修对
→ 跑 PASS_TO_PASS → 全过？没改坏

多维分层 Reward：
  🟦 规则型(权重≥0.4 anchor)：F2P/P2P通过率 + diff命中 + 工具合法性
  🟩 过程型：每轮 critic 评分 + 从结果反推的逐步贡献
  🟨 生成式：小 GenRM 识别语法/逻辑/边界/接口 4 类缺陷（权重<0.3 防偏见）
→ reward 标量（+逐步信号）
```

> **多维分层 Reward 详解**

**为什么要分层？** 只用「测试过/不过」这一个二值信号有三个问题：① **太稀疏**——一条 30 轮的轨迹只在最后给一个 0/1，中间哪步对哪步错完全没信号，长 horizon 难学；② **易被 hack**——模型可能改测试、写 `assert True`、catch 掉异常骗过测试；③ **没有「部分对」**——改对一半也是 0，梯度浪费。所以叠三层，各司其职：

**🟦 规则型（anchor，权重≥0.4）—— 客观、防作弊的「地基」**

| 子项 | 算什么 | 作用 |
|---|---|---|
| **F2P 通过率** | FAIL_TO_PASS 测试通过比例 | 确认 bug 真修好了（疗效） |
| **P2P 通过率** | PASS_TO_PASS 测试仍通过比例 | 确认没引入回归（副作用） |
| **diff 命中** | 生成 patch 与 gold patch 的文件/行重叠度 | 给「改对了文件但测试没跑全」一点部分分 |
| **工具合法性** | 工具调用格式是否合法、参数是否越界 | 惩罚乱调工具 / 幻觉工具 |

- **为什么是 anchor、权重要≥0.4**：这层全部基于**沙箱里可复现的客观事实**（测试真跑、diff 真比），无法靠嘴硬骗过，所以让它占主导权重（≥0.4），作为整个 reward 的「定盘星」。后两层只能在它之上做微调，不能喧宾夺主——这是**防 reward hacking 的核心机制**。

**🟩 过程型 —— 把稀疏终态拆成「每轮都有分」的稠密信号**

- **每轮 critic 评分**：一个 critic 给 agent 的**每一轮动作**打分，比如「这轮 grep 定位到了相关文件 +」「这轮瞎改无关代码 −」。解决「30 轮只有 1 个终态分」的稀疏问题。

  **critic 有哪几种做法（由便宜、稳到贵、虚）**：

  | 方式 | 怎么实现 | 特点 |
  |---|---|---|
  | **纯规则启发式** | 写死的判断：这轮有没有定位到 gold patch 涉及的文件？有没有真跑测试？是不是原地打转（重复同样命令）？编辑的是不是无关文件？ | 0 成本、可复现、不被 hack；但只能抓粗粒度 |
  | **环境信号反推** | 直接拿沙箱反馈当分：这轮命令 exit code=0？grep 有没有命中？编辑后通过的测试数有没有变多？ | 客观、零额外模型 |
  | **进度函数（progress/potential）** | 定义「离解决还差多远」的标量 $m_t$（如已通过测试数/目标数），每轮取**增量** $m_t-m_{t-1}$ | 天然稠密、信用分配清晰 |
  | **小判别模型（PRM/value head）** | 训一个小模型给每轮 state 打分 | 能学细粒度好坏；要训、可能被 hack |
  | **LLM-as-critic（小模型）** | 用小 LLM 读这轮 (思考+动作+观察) 判「这步有无帮助」 | 灵活、能判语义；慢、贵、有偏见 |

  > 实战里最常用、最稳的是**前三种（规则 + 环境信号 + 进度函数）**——客观、便宜、不被骗；LLM-as-critic 只在前三种抓不到的语义维度上少量用。

- **从结果反推的逐步贡献（信用分配）**：拿到最终成功/失败后，**回溯**判断每一步对结果的贡献——哪一步是「转折点」（定位到真正的 bug 行）、哪一步是「弯路」。等于把「整条轨迹一个标量」细化到「每一轮一个信号」，正好喂给阶段 4 的 step-level advantage。
- **作用**：长 horizon 任务（修 bug 要几十轮）能学到「每一步该怎么走」，而不只是「最后成没成」。

**🟨 生成式（GenRM）—— 补「测试覆盖不到」的质量维度，但权重压低（<0.3）**

- **是什么**：一个**小的生成式奖励模型（GenRM）**，读模型产出的代码，识别测试用例覆盖不到的缺陷，分 4 类：**语法**（能跑但写法糟）、**逻辑**（边角逻辑错但测试没测到）、**边界**（空输入/越界没处理）、**接口**（签名/契约不规范）。
- **为什么权重必须 <0.3（防偏见）**：GenRM 是**模型打分**，本身有偏见、可能被 hack（模型学会写「看起来高级」但其实没用的代码去讨好 GenRM）。所以**严格压低权重**，只让它做「锦上添花的微调」，绝不能盖过 🟦 客观测试层。这就是「**可验证的用规则、不可验证的用 GenRM，且规则当 anchor**」原则。

  **GenRM 有哪几种做法**：

  | 方式 | 说明 |
  |---|---|
  | **自训小 GenRM**（最常见） | 拿 4B~7B 小模型，在「代码 + 缺陷标注」数据上微调，输出缺陷类型/质量分 |
  | **复用基座做 judge** | 不单独训，直接用基座 / 一个固定 checkpoint，加 prompt 当 judge |
  | **生成式打分（带 CoT）** | 让模型先生成一段评审推理再给分，比直接出 scalar 更可靠、可解释 |
  | **rubric / pairwise** | 给定评分细则（语法/逻辑/边界/接口四维）逐项判，再加权 |
  | **闭源大模型 API（GPT/Claude）** | 用强模型当 judge，质量高但有成本和依赖 |

  > **GenRM 用 Claude/GPT API 靠谱吗？** —— **离线造数据/评测可以，在线 RL 训练当 reward 不推荐**。原因：① 吞吐/成本爆炸（一个 step 要给几十~上百条轨迹的每一步打分，QPS 极高，走外部 API 会成为训练瓶颈，账单也贵）；② 延迟+限流不可控，会卡住 rollout 破坏 on-policy 节奏；③ 闭源模型随时更新版本，reward 跟着漂移、实验不可复现；④ 模型会学会「讨好 Claude」且其偏好你无法审计/修正（加剧 reward hacking）；⑤ 把训练中的代码轨迹发给第三方 API 常踩合规红线。**通行做法**：在线 RL 的 GenRM 用**自部署小模型**（4B~7B，本地 vLLM/sglang，同集群调用）——低延迟、可控、可复现、能自己迭代；Claude/GPT 留给**离线造数据、蒸馏教师、评测裁判**。

**三层怎么合成一个标量？** 大致是 `reward = w1·规则型 + w2·过程型 + w3·生成式`，其中 `w1≥0.4`（anchor 主导）、`w3<0.3`（GenRM 压低），过程型则额外产出**逐步信号**喂给 step-level advantage。一句话记忆：**规则型保证「不被骗」、过程型保证「学得动（稠密）」、生成式保证「有品味（但别越权）」**。

**阶段 4：Advantage 估计（GRPO 无 critic，用组内相对比较）**

```
episode-level：同 prompt 的 8 条轨迹 reward 组内归一化 → 比平均好的 adv>0
step-level（解决长 horizon 归因）：
   按相同 (state, action) 锚点跨轨迹分组，比较「相似状态下相似动作」哪条更好
   → 把信用分配从「整条轨迹一个标量」细化到「每一轮一个信号」
```

**阶段 5：策略更新（算 loss 反传）**

```
GRPO loss = advantage 加权策略梯度 + KL 正则

长轨迹 / MoE 稳定性（这步关键）：
  · 序列级重要性比率：token级ratio方差大时切「长度归一化几何均值」
                     长轨迹训练崩溃率 23%→4%
  · KL-Cov：只截断 top-k% 高偏移 token 的更新，防熵坍塌但保留探索
  · KL 锚：与 Reference 算 KL，防策略漂移太远（λ 退火 0.5→0.1）

→ Actor 反向传播更新权重
```

**阶段 6：权重同步**

```
Actor 更新完 → 权重同步给 vLLM Rollout 引擎
→ 下个 step 的 rollout 用新权重（持续 on-policy）→ 回到阶段 1
```

### 5.4 整张训练循环图

```
┌─────────────────── 一个 RL Step ───────────────────┐
│  ①取batch ──→ ②Rollout(scaffold多轮+代理拦截)        │
│      ↑              │ 产出patch                      │
│  ⑥同步权重     ③跑测试→Reward(F2P/P2P+多维)          │
│      ↑              │                                │
│  ⑤更新Actor ←── ④Advantage(episode+step双层)         │
│   (序列级比率+KL-Cov+KL锚)                            │
└──────────────── 循环数千 step ──────────────────────┘
```

### 5.5 四阶段课程（外层宏观调度，由易到难）

```
S1 Reasoning   → S2 Tool-Use   → S3 SWE长Horizon → S4 防退化
数学/算法题       短工具任务(≤5轮)  真实仓库修bug(核心)  掺通用对话防遗忘
DAPO四件套        Hybrid Reward    渐进Horizon+多维R   on-policy KL锚
                  +合法性Gate      step级信用分配
```

每个阶段都跑 5.3 那个 6 步循环，只是任务难度、Horizon 上限、reward 复杂度逐阶段加码。

### 5.6 RL 工程要点（容易踩的坑）

| 坑 | 现象 | 解法 |
|---|---|---|
| 长轨迹阻塞 | 一条 100 轮轨迹卡住，整 batch 等它 | **完全异步**：rollout 和训练解耦，轨迹完成即入队、满 batch 即训，GPU 利用率接近 100% |
| 训练崩溃 | 长轨迹 token 级 ratio 方差大、loss 爆 | 序列级重要性比率（几何均值），崩溃率 23%→4% |
| reward hacking | reward 涨但实际没修对（钻空子） | 规则型测试 reward 当 anchor（权重≥0.4），生成式 reward 权重压低 |
| 熵坍塌 | 训到中后期探索退化成「复读同一套路」，泛化崩 | KL-Cov 分位数截断 + 熵 <阈值 时动态提温 |
| 环境作弊 | 模型操控不存在的设备 / 文件骗分 | 运行时审计：非法操作转确定性负反馈（audit 扣分） |

---

## 六、一张图串起全流程

```
通用 Base
   │
   ▼ ① Pre-train（海量代码，CLM+FIM+Repo拓扑序，去污染）
Code Base（会写代码、会补全、懂仓库）
   │
   ▼ ② Mid-train（退火高质量 + 注入工具调用 + 长上下文外推）
增强 Base（懂工具格式、长窗口）
   │
   ▼ ③ SFT（教师成功轨迹，按域分治，工具masking + 分流 + packing）
会按 agent 格式多轮干活的模型
   │
   ▼ ④ RL（沙箱+scaffold+代理拦截，测试当reward，GRPO+课程）
Coder Model（自己能把没见过的 bug 修对）
```

**核心记忆点**：

- **数据**：Pre-train/Mid-train 是纯文本 next-token；SFT 是**带答案的多轮轨迹**（工具返回 masked）；RL 是**只有题面+测试**（轨迹模型自己 rollout）。
- **SFT = 模仿**（学格式与行为），**RL = 试错**（用测试反馈真正提升成功率）。
- **RL 环境三件套**：训练侧（Actor/Rollout/Reference）+ 环境侧（沙箱/scaffold/代理层）+ 数据侧（题面+测试）。
- **on-policy 核心 trick**：代理层劫持 scaffold 的 LLM 调用 → 转发到待训 vLLM，沙箱跑测试当 reward。
- **稳定性三件套**：序列级重要性比率（防崩溃）+ KL-Cov（防熵坍塌）+ KL 锚（防漂移）。
