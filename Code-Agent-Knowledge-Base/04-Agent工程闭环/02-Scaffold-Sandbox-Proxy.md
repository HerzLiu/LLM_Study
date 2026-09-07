---
tags: [Code-Agent, Scaffold, Sandbox, Proxy, On-Policy]
status: Active
source: "用户提供：训练Coder模型全流程.md §五.1-五.3"
stage: engineering
---

# Scaffold / Sandbox / Proxy

⬅ [01-Agent工程闭环总览](01-Agent%E5%B7%A5%E7%A8%8B%E9%97%AD%E7%8E%AF%E6%80%BB%E8%A7%88.md) | ➡ [03-工具调用与轨迹Schema](03-%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E4%B8%8E%E8%BD%A8%E8%BF%B9Schema.md)

## 🎬 故事比喻：导演、摄影棚和电话总机

Scaffold 像导演：安排下一幕该让主角做什么、什么时候喊停、什么时候重拍。

Sandbox 像摄影棚：所有爆破、撞车、危险动作都在棚里发生，不影响真实世界。

Proxy 像电话总机：演员以为自己在给“模型 API”打电话，但总机会把电话转到当前正在训练的模型，确保每句台词都来自同一个演员。

## Scaffold

Scaffold 是 agent 框架本身，比如 SWE-agent、OpenHands、Claude Code 风格框架。

它负责：

- 拼 prompt。
- 暴露工具。
- 维护 messages / state。
- 执行多轮循环。
- 判断 submit。

## Sandbox

Sandbox 是可复现、隔离的执行环境。

它负责：

- checkout base commit。
- 执行 bash / editor / pytest。
- 限制网络、文件系统、时间、资源。
- 记录审计日志。

## Proxy

Proxy 是 RL 训练里的关键 trick。

Scaffold 以为自己在调 OpenAI / Anthropic API，实际被代理到当前 rollout engine：

```bash
OPENAI_BASE_URL=http://rl-proxy:8080
ANTHROPIC_BASE_URL=http://rl-proxy:8080
```

这样 scaffold 可以当黑盒复用，而每一步动作仍来自正在训练的 policy。

## 为什么重要

没有 proxy，rollout 很容易变成 off-policy：轨迹来自外部模型或旧模型，更新就不再对应当前 policy。

## 🔧 技术拆解：为什么三者要分离

| 分离对象 | 好处 |
|---|---|
| scaffold 和 model 分离 | 可以换模型，不重写 agent loop |
| sandbox 和 host 分离 | 可以并发、可复现、可清理 |
| proxy 和 scaffold 分离 | 可以复用现有工具协议，同时保证 on-policy |

## 🧪 例子：一次工具调用的路径

```text
assistant 生成:
  {"tool": "bash", "args": "pytest tests/test_timeout.py -q"}

scaffold:
  校验工具格式和权限

sandbox:
  在隔离容器里执行 pytest

observation:
  返回短日志、退出码、失败测试名

proxy:
  下一轮把 observation 拼回 prompt，继续调当前 rollout 模型
```

## ⚠️ 避坑

- sandbox 镜像版本不固定，reward 会抖。
- proxy 协议翻译不严谨，会把训练样本格式搞乱。
- scaffold 偷偷调用外部 LLM，会破坏 on-policy。

## 回链

- [07-Repo地图（SWE-agent-OpenHands等）](../../agentic-rl-knowledge-base/04-%E7%AC%AC4%E7%AB%A0-Code-Agentic-RL%E4%B8%93%E9%A2%98/07-Repo%E5%9C%B0%E5%9B%BE%EF%BC%88SWE-agent-OpenHands%E7%AD%89%EF%BC%89.md)
- [04-物理沙箱与Hooks](../../Claude-Code-Source/07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/04-%E7%89%A9%E7%90%86%E6%B2%99%E7%AE%B1%E4%B8%8EHooks.md)
