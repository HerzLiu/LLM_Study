---
tags: [Code-Agent, 数据工程, Pretrain, FIM, Repo-Level]
status: Active
source: "用户提供：训练Coder模型全流程.md §一.1"
stage: data
---

# Pre-train 数据工程

⬅ [00-四阶段数据总览](00-%E5%9B%9B%E9%98%B6%E6%AE%B5%E6%95%B0%E6%8D%AE%E6%80%BB%E8%A7%88.md) | ➡ [02-Midtrain数据工程](02-Midtrain%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B.md)

## 目标

让通用 base 模型变成懂代码的 code base：会语法、API、常见工程模式、文档语言和跨文件引用。

## 喂什么数据

| 数据类 | 作用 |
|---|---|
| 原始代码 | 学多语言语法、库 API、模式 |
| Repo-level 代码 | 学 import、调用链、跨文件引用 |
| 代码相关文本 | 学自然语言和代码对齐 |
| 通用网页文本 | 保留通用语言能力 |
| 数学 / 推理 | 补代码推理能力 |

## 两个 Code 专属技巧

### FIM

FIM 把代码切成 `prefix / middle / suffix`，再把 middle 放到末尾让模型预测。它让模型学会“看前后文补中间”。

重点：loss 仍是标准 next-token CE，特殊性在数据重排。

### Repo-level 拓扑序

把同一仓库多文件按依赖顺序拼起来，用 `<|file_sep|>` 分隔。被依赖文件排前面，模型预测调用方时已经见过定义。

重点：这让跨文件引用成为有效监督信号。

## 训练信号

标准 causal LM next-token cross entropy。

## 和已有库的回链

- LLM 预训练基础：[05-Pretrain预训练](../../Happy-LLM/04-%E7%AC%AC4%E7%AB%A0-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/05-Pretrain%E9%A2%84%E8%AE%AD%E7%BB%83.md)
- next-token loss 前置：[02-从交叉熵到next-token-loss](../../ml-dl-foundations/07-%E7%AC%AC7%E7%AB%A0-%E6%A1%A5%E6%8E%A5%E5%88%B0%E4%BD%A0%E5%B7%B2%E5%AD%A6%E7%9A%84%E4%B8%96%E7%95%8C/02-%E4%BB%8E%E4%BA%A4%E5%8F%89%E7%86%B5%E5%88%B0next-token-loss.md)

## 自测问题

- FIM 为什么能提升 IDE 补全能力？
- repo-level 随机顺序拼接为什么会制造噪声？
- Pre-train 阶段为什么要做评测集去污染？

