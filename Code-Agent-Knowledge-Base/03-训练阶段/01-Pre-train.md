---
tags: [Code-Agent, 训练阶段, Pretrain, FIM, Repo-Level]
status: Active
source: "用户提供：训练Coder模型全流程.md §二"
stage: pretrain
---

# Pre-train

⬅ [04-RL任务与验证器数据](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/04-RL%E4%BB%BB%E5%8A%A1%E4%B8%8E%E9%AA%8C%E8%AF%81%E5%99%A8%E6%95%B0%E6%8D%AE.md) | ➡ [02-Mid-train](02-Mid-train.md)

## 🎬 故事比喻：让实习生先读完维修手册

在让一个人进车间修车之前，先让他读维修手册、零件目录、历史维修记录。

Pre-train 就是这个阶段：还不要求模型真的接工单，也不要求它跑测试，只要求它在海量代码和文档里学会“代码世界的语言”。

## 目标

把通用 base 变成 **Code Base**：会写代码、会补全、懂基本仓库结构。

## 喂什么数据

参考 [01-Pretrain数据工程](../02-%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B/01-Pretrain%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B.md)：

- 海量多语言代码。
- repo-level 多文件拼接。
- README、文档、教程、代码问答。
- 通用高质量网页文本。
- 数学与推理语料。

## 🔧 技术拆解：训练信号

标准 next-token cross entropy。

```text
输入: def normalize_path(path):
目标: 预测后续 token
```

FIM 和 repo-level 拓扑序看起来很特殊，但 loss 仍是 CE。真正变的是数据排列。

## 关键工程 trick

- 多语言配比不能只看 GitHub star，要考虑每种语言的可学习收益。
- CLM、FIM、repo-level 拓扑序可以混合训练。
- 做 13-gram 等去重，防止 HumanEval、SWE-bench 等评测污染。

## 🧪 例子：repo-level 拓扑序为什么重要

```text
<|file_sep|>utils/timeout.py
def coerce_timeout(timeout):
    ...

<|file_sep|>requests/adapters.py
from utils.timeout import coerce_timeout
```

如果 `utils/timeout.py` 先出现，模型在预测 `adapters.py` 的调用时，上文真的有定义；如果随机顺序反过来，跨文件监督就变弱甚至变成噪声。

## ⚠️ 工程坑

| 坑 | 后果 | 处理 |
|---|---|---|
| 评测污染 | benchmark 虚高 | n-gram / repo / commit 级去重 |
| 只训单文件 | 跨文件定位弱 | repo-level 拼接 |
| 忽略自然语言文档 | issue -> code 对齐弱 | README / docs / Q&A 混入 |
| FIM 比例不当 | 补全能力和续写能力失衡 | 控制 FIM / CLM 混比 |

## 产物

Code Base：

- 懂代码 token 分布。
- 有基本补全能力。
- 见过跨文件依赖。
- 还不会稳定地按 agent 协议多轮调工具。

## 回链

- [05-Pretrain预训练](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)
- [09-预训练循环](../../Happy-LLM/05-%E7%AC%AC5%E7%AB%A0-%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B/09-%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- [02-从交叉熵到next-token-loss](../../ml-dl-foundations/07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md)

## ❓ 自测问题

- Pre-train 的产物为什么还不是 Code Agent？
- 去污染为什么对代码模型尤其重要？
- FIM 与 repo-level 数据分别服务哪类能力？
