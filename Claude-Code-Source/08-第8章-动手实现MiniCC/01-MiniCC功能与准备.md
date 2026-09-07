---
tags: [Claude-Code, 第8章, 准备, 功能清单]
chapter: 8
section: 8.1
---

# 8.1 MiniCC 功能 + 环境准备

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-完整代码上半](02-%E5%AE%8C%E6%95%B4%E4%BB%A3%E7%A0%81%E4%B8%8A%E5%8D%8A.md)

---

## 🎯 MiniCC 功能清单

| 功能 | 来自哪一章 | 实现位置 |
|---|---|---|
| **主循环** | 第 2 章 | `run_agent()` |
| **工具：Read** | 第 3 章 | `READ_TOOL` |
| **工具：Write** | 第 3 章 | `WRITE_TOOL` |
| **工具：Edit** | 第 3 章 | `EDIT_TOOL` |
| **工具：Bash** | 第 3 章 | `BASH_TOOL` |
| **工具：Glob** | 第 3 章 | `GLOB_TOOL` |
| **上下文管理** | 第 4 章 | `auto_compact()` + `estimate_tokens()` |
| **System Prompt** | 第 5 章 | `build_system_prompt()` |
| **4 种 Permission Mode** | 第 7 章 | `PermissionMode` enum |
| **危险模式检测** | 第 7 章 | `DANGEROUS_PATTERNS` |
| **权限弹窗** | 第 7 章 | `check_permission()` |
| **CLI 交互** | — | `main()` REPL |

---

## 🛠 环境准备

### 1. 创建虚拟环境（推荐）

```bash
cd ~/llm-study
python3 -m venv venv
source venv/bin/activate           # macOS/Linux
# Windows: venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install anthropic rich
```

| 包 | 干嘛 |
|---|---|
| `anthropic` | 官方 Python SDK，调 Claude API |
| `rich` | 终端漂亮渲染（颜色、表格、markdown） |

### 3. 准备 API Key

去 https://console.anthropic.com 注册账号 → 拿一个 API key（充值至少 $5 才能用）。

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

> 💡 建议把这行写到 `~/.zshrc` 或 `~/.bashrc`，下次自动加载。

### 4. （可选）选 Sonnet 还是 Haiku

```bash
# 用 Haiku 跑（便宜 12 倍）
export MINICC_MODEL="claude-haiku-4-5"

# 默认 Sonnet
# unset MINICC_MODEL
```

---

## 📂 项目结构

```
~/llm-study/minicc/
└── minicc.py        # 主程序（单文件 ~400 行）
```

为了方便初学者跟读，**单文件版本**。生产化时再拆分。

---

## ⚠️ 没 API key 怎么办

如果暂时没有 key 也能学：

### 方案 A：mock LLM

把 `client.messages.create(...)` 换成假实现：

```python
class FakeAnthropic:
    @property
    def messages(self):
        return self
    def create(self, **kwargs):
        # 简单脚本化响应
        return MockResponse(content=[
            MockBlock(type="text", text="假回答")
        ])
```

### 方案 B：用 Ollama 本地模型

[Ollama](https://ollama.ai) 跑本地模型，写个适配层把请求转给它。

### 方案 C：先读代码不跑

第 8.2、8.3 节会逐段讲，**不跑也能学到 90%**。

---

## 📋 下一步行动清单

```
1. 装好环境
   - [ ] python3 venv
   - [ ] pip install anthropic rich
   - [ ] 设置 ANTHROPIC_API_KEY（或选 mock）

2. 看代码
   - [ ] 读 [[02-完整代码上半]] 配置和工具
   - [ ] 读 [[03-完整代码下半]] 主循环和 CLI

3. 跑起来
   - [ ] 跑通 [[04-运行与练习]] 第一个 prompt
   - [ ] 试 /plan, /default, /bypass 各模式
   - [ ] 触发一次危险检测

4. 改造
   - [ ] 选 1 个练习深度实现
```

---

## 🔗 延伸阅读

- 上一节：[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节：[02-完整代码上半](02-%E5%AE%8C%E6%95%B4%E4%BB%A3%E7%A0%81%E4%B8%8A%E5%8D%8A.md)
- 完整源码：`~/llm-study/minicc/minicc.py`
- 真 Claude Code：`~/llm-study/_source/`

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)　|	➡ [02-完整代码上半](02-%E5%AE%8C%E6%95%B4%E4%BB%A3%E7%A0%81%E4%B8%8A%E5%8D%8A.md)
