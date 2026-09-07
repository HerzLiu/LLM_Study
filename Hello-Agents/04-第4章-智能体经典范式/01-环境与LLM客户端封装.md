---
tags: [Hello-Agents, 第4章, 实战准备, LLM客户端]
chapter: 4
section: 4.1
---

# 4.1 环境准备与 LLM 客户端封装

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-ReAct范式](02-ReAct%E8%8C%83%E5%BC%8F.md)

> 在动手三大范式之前,先**搭好开发环境** + **封装一个统一的 LLM 客户端**,后续三章共用。

---

## 🎬 故事比喻:做菜前要先备好厨房

要做三道大菜(ReAct / Plan-and-Solve / Reflection),
得先:
- **买齐食材**(安装依赖)
- **装好刀具**(配置 API)
- **做好基础酱料**(封装 LLM 客户端)

→ 这一节就是**备菜阶段**。

---

## 🛠 4.1.1 安装依赖

```bash
pip install openai python-dotenv
```

| 库 | 用途 |
|---|---|
| `openai` | OpenAI 标准 API 客户端(兼容大多数 LLM 服务) |
| `python-dotenv` | 从 `.env` 文件加载环境变量 |

**Python 版本**:建议 **3.10+**。

---

## 🔑 4.1.2 配置 API 密钥(`.env` 文件)

在项目根目录创建 **`.env`** 文件:

```bash
# .env
LLM_API_KEY="YOUR-API-KEY"
LLM_MODEL_ID="qwen-plus"      # 或 gpt-4o-mini / deepseek-chat 等
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_TIMEOUT="60"
```

### 几种常见服务的配置

| 服务 | `LLM_BASE_URL` | 备注 |
|---|---|---|
| **OpenAI 官方** | `https://api.openai.com/v1` | 需 VPN |
| **阿里通义** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 国内稳定 |
| **DeepSeek** | `https://api.deepseek.com/v1` | 性价比高 |
| **硅基流动** | `https://api.siliconflow.cn/v1` | 聚合多家开源模型 |
| **本地 Ollama** | `http://localhost:11434/v1` | 本地部署 |

→ **OpenAI 兼容接口**已成行业标准,**改个 URL 就能换服务**。

> ⚠️ **`.env` 千万别提交到 GitHub**(在 `.gitignore` 里加上 `.env`)

---

## 🐍 4.1.3 封装 LLM 客户端(本书共用)

> 这个 `HelloAgentsLLM` 类**后续三大范式都会用**,封装好一次,无限复用。

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的 LLM 客户端
    特点:
    - 兼容任何 OpenAI 接口的服务
    - 默认流式输出(边生成边打印,体验好)
    - 配置从 .env 自动加载
    """

    def __init__(self, model: str = None, apiKey: str = None,
                 baseUrl: str = None, timeout: int = None):
        """
        初始化:优先用传入参数,否则从 .env 加载
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError(
                "模型ID、API密钥和服务地址必须被提供或在 .env 文件中定义。"
            )

        self.client = OpenAI(
            api_key=apiKey, base_url=baseUrl, timeout=timeout
        )

    def think(self, messages: List[Dict[str, str]],
              temperature: float = 0) -> str:
        """
        调用 LLM 思考,返回响应文本
        - messages: OpenAI 标准消息格式 [{"role": ..., "content": ...}, ...]
        - temperature: 默认 0(Agent 用确定性输出)
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,   # ⭐ 流式输出
            )

            # 处理流式响应,边收边打印
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()   # 流式结束后换行

            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用 LLM API 时发生错误: {e}")
            return None
```

### 📝 代码解读

#### 关键设计 1:**配置优先级**

```python
self.model = model or os.getenv("LLM_MODEL_ID")
```
- 优先用**传入参数**(灵活,可临时切换)
- 没传则从**环境变量**读(默认配置)

→ 这种"**参数覆盖环境变量**"模式是 SDK 设计标准做法。

#### 关键设计 2:**流式输出**(stream=True)

```python
response = self.client.chat.completions.create(
    ..., stream=True,
)
for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

**为什么用流式?**
- LLM 生成慢,流式让用户**边生成边看到**(体验好)
- 调试 Agent 时**实时看到 LLM 在想什么**

#### 关键设计 3:**Agent 默认 `temperature=0`**

```python
def think(self, messages, temperature: float = 0):
```

→ Agent 需要**稳定可预测**的输出,默认 0。
→ 详见 [3.2.1 Temperature](../03-%E7%AC%AC3%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/02-%E4%B8%8ELLM%E4%BA%A4%E4%BA%92.md#%E6%97%8B%E9%92%AE-1temperature%E6%B8%A9%E5%BA%A6)

#### 关键设计 4:**OpenAI 标准消息格式**

```python
messages = [
    {"role": "system", "content": "你是 Python 专家"},
    {"role": "user", "content": "写快速排序"},
]
```

| role | 用途 |
|---|---|
| `system` | 系统指令(定义角色和约束) |
| `user` | 用户输入 |
| `assistant` | LLM 之前的回复(多轮对话用) |

---

## 🔬 测试客户端

```python
if __name__ == '__main__':
    llmClient = HelloAgentsLLM()

    exampleMessages = [
        {"role": "system", "content": "你是一个 Python 编程助手。"},
        {"role": "user", "content": "写一个快速排序算法"},
    ]

    print("--- 调用 LLM ---")
    responseText = llmClient.think(exampleMessages)
    print(f"\n\n--- 完整响应 ---\n{responseText}")
```

成功跑通,会看到流式输出的快排代码。

---

## 📦 后续章节的统一假设

后续 4.2 / 4.3 / 4.4 三节,**都假设**:
- 你已经创建 `.env` 文件
- `HelloAgentsLLM` 类已经定义好
- 直接 `llm_client = HelloAgentsLLM()` 就能用

→ **基础设施搭好,专注 Agent 范式实现**。

---

## ⚠️ 小白避坑

1. **`.env` 一定要在项目根目录**
   - `load_dotenv()` 默认从当前目录找
   - 错位置 → 加载失败
2. **API Key 别硬编码到代码里**
   - 改 `.env` 才是正道
   - GitHub 公开仓库 = 千万别提交 `.env`
3. **OpenAI 兼容服务的细微差异**
   - 不同服务可能不支持某些参数(如 `seed`)
   - 出错先看官方文档
4. **`timeout` 设得太短会失败**
   - LLM 生成慢,**60 秒起步**

---

## 📌 4.1 节要点

| 知识点 | 一句话 |
|---|---|
| `.env` 配置 | API Key + Model ID + Base URL,**3 件套** |
| `HelloAgentsLLM` | 本书统一 LLM 客户端,后续都用 |
| `temperature=0` | Agent 默认确定性输出 |
| 流式输出 | 用户体验好 + 调试方便 |
| OpenAI 兼容接口 | **行业标准**,改 URL 即可换服务 |

---

## 🔗 延伸阅读

- 上一节:[00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)
- 下一节:[02-ReAct范式](02-ReAct%E8%8C%83%E5%BC%8F.md) ⭐ **第一个 Agent 范式**
- 采样参数细节:[02-与LLM交互](../03-%E7%AC%AC3%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/02-%E4%B8%8ELLM%E4%BA%A4%E4%BA%92.md)

---

⬅ [00-章节总览](00-%E7%AB%A0%E8%8A%82%E6%80%BB%E8%A7%88.md)	|	➡ [02-ReAct范式](02-ReAct%E8%8C%83%E5%BC%8F.md)
