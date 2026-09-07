---
tags: [Claude-Code, 第8章, 完整代码, MiniCC]
chapter: 8
section: 8.99
---

# 8.99 完整代码文件位置

⬅ [04-运行与练习](04-%E8%BF%90%E8%A1%8C%E4%B8%8E%E7%BB%83%E4%B9%A0.md)　|	➡ [附录 →](../%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)

---

## 📂 文件位置

```
~/llm-study/minicc/
└── minicc.py    (~568 行单文件)
```

在 Obsidian 里可直接打开链接：
- 完整代码见本章「完整代码上半」与「完整代码下半」笔记；原独立脚本未收录。
- 或终端：`code ~/llm-study/minicc/minicc.py`

---

## 🗺️ 代码地图

```
minicc.py
├── 1. imports                                    (1-30)
├── 2. CONFIG 常量                                 (32-45)
├── 3. PermissionMode enum                         (47-52)
├── 4. DANGEROUS_PATTERNS + check_dangerous       (55-72)
├── 5. Tool dataclass                              (74-100)
├── 6. 工具实现
│   ├── read_handler + READ_TOOL                   (102-145)
│   ├── write_handler + WRITE_TOOL                 (147-180)
│   ├── edit_handler + EDIT_TOOL                   (182-220)
│   ├── bash_handler + BASH_TOOL                   (222-265)
│   └── glob_handler + GLOB_TOOL                   (267-290)
├── 7. ALL_TOOLS / TOOLS_BY_NAME                   (293-296)
├── 8. filter_tools_by_mode + check_permission     (300-340)
├── 9. build_system_prompt                         (343-385)
├── 10. estimate_tokens + auto_compact             (388-425)
├── 11. blocks_to_dict 辅助                        (428-445)
├── 12. ⭐ run_agent 主循环                         (448-525)
└── 13. CLI: print_help + main                     (528-568)
```

---

## 🚀 快速启动

```bash
# 1. 装依赖
pip install anthropic rich

# 2. 设 API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. 跑！
python3 ~/llm-study/minicc/minicc.py
```

详细体验见 [04-运行与练习](04-%E8%BF%90%E8%A1%8C%E4%B8%8E%E7%BB%83%E4%B9%A0.md)。

---

## 📖 代码到章节映射

每一段代码都能在前面 7 章找到对应的"理论"：

| 代码段 | 理论章节 |
|---|---|
| `PermissionMode` enum | [01-5种权限模式](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/01-5%E7%A7%8D%E6%9D%83%E9%99%90%E6%A8%A1%E5%BC%8F.md) |
| `DANGEROUS_PATTERNS` | [03-Bash危险模式](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/03-Bash%E5%8D%B1%E9%99%A9%E6%A8%A1%E5%BC%8F.md) |
| `Tool` dataclass | [02-Tool接口设计](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/02-Tool%E6%8E%A5%E5%8F%A3%E8%AE%BE%E8%AE%A1.md) |
| `READ_TOOL` | [03-FileReadTool解剖](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/03-FileReadTool%E8%A7%A3%E5%89%96.md) |
| `EDIT_TOOL` 唯一匹配 | [06-设计哲学5条](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md) |
| `is_concurrency_safe=False` | [06-设计哲学5条](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/06-%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A65%E6%9D%A1.md) |
| `filter_tools_by_mode` | [05-注册过滤执行](../03-%E7%AC%AC3%E7%AB%A0-%E5%B7%A5%E5%85%B7%E7%B3%BB%E7%BB%9F/05-%E6%B3%A8%E5%86%8C%E8%BF%87%E6%BB%A4%E6%89%A7%E8%A1%8C.md) |
| `check_permission` 5 层 | [02-canUseTool过滤器](../07-%E7%AC%AC7%E7%AB%A0-%E5%AE%89%E5%85%A8%E4%B8%8E%E6%9D%83%E9%99%90/02-canUseTool%E8%BF%87%E6%BB%A4%E5%99%A8.md) |
| `build_system_prompt` | [01-入职手册的13节](../05-%E7%AC%AC5%E7%AB%A0-System-Prompt%E5%B7%A5%E7%A8%8B/01-%E5%85%A5%E8%81%8C%E6%89%8B%E5%86%8C%E7%9A%8413%E8%8A%82.md) |
| `auto_compact` | [02-AutoCompact压缩](../04-%E7%AC%AC4%E7%AB%A0-%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86/02-AutoCompact%E5%8E%8B%E7%BC%A9.md) |
| `run_agent` while 循环 | [02-queryLoop骨架](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/02-queryLoop%E9%AA%A8%E6%9E%B6.md) |
| `MAX_TURNS` | [05-工业级护栏](../02-%E7%AC%AC2%E7%AB%A0-Agent-Loop%E4%B8%BB%E5%BE%AA%E7%8E%AF/05-%E5%B7%A5%E4%B8%9A%E7%BA%A7%E6%8A%A4%E6%A0%8F.md) |

---

## 🎯 学习闭环

```
读理论章节
   ↓
看 MiniCC 对应代码
   ↓
跑起来
   ↓
改一个工具 / 加一个机制
   ↓
回头再读理论
   ↓
更深的理解
```

---

## 🔗 延伸阅读

- 上一节：[04-运行与练习](04-%E8%BF%90%E8%A1%8C%E4%B8%8E%E7%BB%83%E4%B9%A0.md)
- 全书结语：[00-总览](../00-%E6%80%BB%E8%A7%88.md)
- 进入附录：[术语表](../%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)

---

⬅ [04-运行与练习](04-%E8%BF%90%E8%A1%8C%E4%B8%8E%E7%BB%83%E4%B9%A0.md)　|	➡ [附录 →](../%E9%99%84%E5%BD%95/%E6%9C%AF%E8%AF%AD%E8%A1%A8.md)
