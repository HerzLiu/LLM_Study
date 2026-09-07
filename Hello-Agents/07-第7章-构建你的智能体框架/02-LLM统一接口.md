---
tags: [Hello-Agents, 第7章, HelloAgentsLLM, 多提供商, 本地模型]
chapter: 7
section: 7.2
---

# 7.2 HelloAgentsLLM 扩展(多提供商 + 本地模型 + 自动检测)

⬅ [01-为什么自建框架](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E8%87%AA%E5%BB%BA%E6%A1%86%E6%9E%B6.md)	|	➡ [03-框架核心接口](03-%E6%A1%86%E6%9E%B6%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3.md)

---

## 🎬 故事比喻:万能遥控器

第 4 章我们做的 `HelloAgentsLLM` 像个**专用遥控器** —— 只能控制一台电视。
本节我们升级为**万能遥控器**:
- 自动识别品牌(OpenAI / ModelScope / 智谱)
- 支持本地设备(VLLM / Ollama)
- 一个按钮,啥都能用

---

## 🎯 7.2.1 升级目标(3 个)

| 目标 | 解释 |
|---|---|
| **多提供商支持** | OpenAI / ModelScope / 智谱 AI **无缝切换** |
| **本地模型集成** | VLLM + Ollama 生产级方案 |
| **自动检测机制** | 框架自动推断 provider,**用户零配置** |

---

## 🔧 7.2.2 多提供商支持:`provider` 参数

### 问题

之前的 `HelloAgentsLLM` 只能通过 `api_key + base_url` 手动配置。
不同服务商有不同的**环境变量名 / 默认地址 / 推荐模型**:

| 服务商 | API Key 环境变量 | Base URL |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| ModelScope | `MODELSCOPE_API_KEY` | `https://api-inference.modelscope.cn/v1/` |
| 智谱 | `ZHIPU_API_KEY` | `https://open.bigmodel.cn/api/paas/v4/` |

→ 每次切换都要查文档改代码。

### 解法:`provider` 参数

```python
# 极简使用
llm = HelloAgentsLLM(provider="modelscope")
# 内部自动:
#   1. 读取 MODELSCOPE_API_KEY 环境变量
#   2. 配置 base_url = https://api-inference.modelscope.cn/v1/
#   3. 设置默认模型 Qwen/Qwen2.5-VL-72B-Instruct
```

---

## 🛠 7.2.3 不修改源码扩展新提供商(继承)

> 直接修改 pip 装的库源码 → **升级会丢失**。
> **正确做法**:**继承 + 重写**。

### 示例:扩展支持 ModelScope

```python
# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    自定义 LLM 客户端,继承增加对 ModelScope 的支持
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # ⭐ 拦截 modelscope,自定义处理
        if provider == "modelscope":
            print("✓ 使用自定义 ModelScope Provider")
            self.provider = "modelscope"
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            if not self.api_key:
                raise ValueError("请设置 MODELSCOPE_API_KEY 环境变量")

            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get("temperature", 0.7)
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        else:
            # ⭐ 其他情况:**完全交给父类**(保留原始功能)
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                **kwargs,
            )
```

### 使用

```python
from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()

llm = MyLLM(provider="modelscope")
messages = [{"role": "user", "content": "你好"}]
response = llm.think(messages)
```

→ **没改源码,扩展了新功能,升级不丢失** —— 这就是**"开闭原则"**(对扩展开放,对修改关闭)。

---

## 🏠 7.2.4 本地模型集成(VLLM + Ollama)

> [第 3 章](../03-%E7%AC%AC3%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80/02-%E4%B8%8ELLM%E4%BA%A4%E4%BA%92.md) 我们用了 HuggingFace Transformers —— 学习够用但**生产性能差**。
> **VLLM + Ollama 才是生产级方案**。

### VLLM(高性能推理库)

**特点**:
- **PagedAttention** 等先进技术
- 吞吐量比 HF Transformers 高数倍
- **兼容 OpenAI API**

**部署 3 步**:

```bash
# 1. 安装(注意 CUDA 版本)
pip install vllm

# 2. 启动 API 服务(自动从 HuggingFace 下载模型)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --host 0.0.0.0 \
    --port 8000

# 3. 服务自动在 http://localhost:8000/v1 提供 OpenAI 兼容 API
```

### Ollama(更简化,一行命令)

**特点**:
- 模型下载 + 配置 + 服务 一体化
- 自动 GPU 加速
- 适合**快速本地实验**

**部署 2 步**:
```bash
# 1. 装客户端(https://ollama.com)

# 2. 一行启动(自动下载模型)
ollama run llama3

# 默认在 http://localhost:11434/v1 提供 OpenAI 兼容 API
```

### 接入 HelloAgentsLLM(超简单)

```python
# 方式 1: 显式指定 provider
llm = HelloAgentsLLM(
    provider="vllm",
    model="Qwen/Qwen1.5-0.5B-Chat",
    base_url="http://localhost:8000/v1",
    api_key="vllm",   # 本地服务可填任意非空字符串
)

# 方式 2: 环境变量 + 自动检测(零代码改动)
# .env 文件:
#   LLM_BASE_URL=http://localhost:8000/v1
#   LLM_API_KEY=vllm
llm = HelloAgentsLLM()   # 自动检测为 vllm
```

```python
# Ollama 同理
llm = HelloAgentsLLM(
    provider="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

→ **统一设计**,**云端 API 和本地模型自由切换** —— 大幅提升部署灵活性。

---

## 🤖 7.2.5 自动检测机制(核心算法)

### 设计原则:**约定优于配置**

不让用户改代码,**框架自动猜你想用啥**。

### 检测优先级(3 级)

```
1. 最高优先级:特定服务商的环境变量
   if MODELSCOPE_API_KEY exists: provider = "modelscope"
   if OPENAI_API_KEY exists:    provider = "openai"
   if ZHIPU_API_KEY exists:      provider = "zhipu"

2. 次高:根据 base_url
   if "api-inference.modelscope.cn" in url: → modelscope
   if "open.bigmodel.cn" in url:            → zhipu
   if "localhost:11434":                     → ollama
   if "localhost:8000":                      → vllm

3. 辅助:API Key 格式
   if key.startswith("ms-"): → modelscope
   ...
```

### 关键代码(简化版)

```python
def _auto_detect_provider(self, api_key, base_url) -> str:
    """自动检测 LLM 提供商"""

    # 1. 检查特定环境变量
    if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
    if os.getenv("OPENAI_API_KEY"):     return "openai"
    if os.getenv("ZHIPU_API_KEY"):       return "zhipu"

    actual_api_key  = api_key  or os.getenv("LLM_API_KEY")
    actual_base_url = base_url or os.getenv("LLM_BASE_URL")

    # 2. 根据 base_url 判断
    if actual_base_url:
        url = actual_base_url.lower()
        if "api-inference.modelscope.cn" in url: return "modelscope"
        if "open.bigmodel.cn" in url:             return "zhipu"
        if "localhost" in url or "127.0.0.1" in url:
            if ":11434" in url: return "ollama"
            if ":8000"  in url: return "vllm"
            return "local"

    # 3. 根据 API Key 格式辅助判断
    if actual_api_key:
        if actual_api_key.startswith("ms-"): return "modelscope"

    return "auto"
```

### 配套:**`_resolve_credentials` 完成配置**

```python
def _resolve_credentials(self, api_key, base_url) -> tuple[str, str]:
    """根据 provider 解析最终的 api_key 和 base_url"""
    if self.provider == "modelscope":
        return (
            api_key  or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY"),
            base_url or os.getenv("LLM_BASE_URL")       or "https://api-inference.modelscope.cn/v1/",
        )
    elif self.provider == "openai":
        return (
            api_key  or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
            base_url or os.getenv("LLM_BASE_URL")    or "https://api.openai.com/v1",
        )
    # ... 其他 provider
```

→ **自动 + 显式 + fallback** 三层保护,鲁棒性强。

---

## 🎯 自动检测实战体验

### 用户视角:**零配置体验**

`.env` 文件:
```
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_ID=llama3
```

Python 代码:
```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

load_dotenv()

# 无需传 provider,自动检测
llm = HelloAgentsLLM()
# 框架日志: 检测到 provider = "ollama"

# 调用方式完全不变
messages = [{"role": "user", "content": "你好"}]
for chunk in llm.think(messages):
    print(chunk, end="")
```

→ **用户根本不用懂 provider 概念,框架替他想好了**。

---

## 📊 HelloAgentsLLM 版本演进

| 维度 | 4.1.3 版(基础) | 7.2 版(增强) |
|---|---|---|
| 多提供商 | ❌ 只支持 OpenAI 兼容 | ✅ **OpenAI/ModelScope/智谱/...** |
| 本地模型 | ❌ | ✅ **VLLM/Ollama** |
| 自动检测 | ❌ | ✅ **基于环境/URL/Key** |
| 用户体验 | 要手动写 base_url | **零配置** |

→ **从简单开始,逐步完善** —— 这就是渐进式设计。

---

## ⚠️ 小白避坑

1. **`provider="auto"` 是默认值**
   - 不传 = 自动检测
   - 明确传值 = 跳过检测
2. **环境变量名要严格**
   - `MODELSCOPE_API_KEY` ≠ `modelscope_api_key`
   - Python `os.getenv` 大小写敏感
3. **本地服务 API Key 可填假的**
   - VLLM/Ollama 不验证
   - 但**不能为空**(OpenAI SDK 会报错)
4. **VLLM 部署要注意 CUDA 版本**
   - CUDA 11.x 和 12.x 对应不同 wheel
   - 装错会出怪问题

---

## 📌 7.2 节要点

| 知识点 | 一句话 |
|---|---|
| **多提供商** | `provider` 参数统一接口 |
| **扩展方式** | **继承 + 重写**(不改源码) |
| **本地模型** | VLLM(高性能) / Ollama(最简) |
| **自动检测** | 环境变量 → URL → Key 格式 三级优先 |
| **设计原则** | **约定优于配置** + **开闭原则** |

---

## 🔗 延伸阅读

- 上一节:[01-为什么自建框架](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E8%87%AA%E5%BB%BA%E6%A1%86%E6%9E%B6.md)
- 下一节:[03-框架核心接口](03-%E6%A1%86%E6%9E%B6%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3.md) —— Message/Config/Agent 基类
- 第 4 章基础版:[4.1.3 LLM 客户端封装](../04-%E7%AC%AC4%E7%AB%A0-%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F/01-%E7%8E%AF%E5%A2%83%E4%B8%8ELLM%E5%AE%A2%E6%88%B7%E7%AB%AF%E5%B0%81%E8%A3%85.md)

---

⬅ [01-为什么自建框架](01-%E4%B8%BA%E4%BB%80%E4%B9%88%E8%87%AA%E5%BB%BA%E6%A1%86%E6%9E%B6.md)	|	➡ [03-框架核心接口](03-%E6%A1%86%E6%9E%B6%E6%A0%B8%E5%BF%83%E6%8E%A5%E5%8F%A3.md)
