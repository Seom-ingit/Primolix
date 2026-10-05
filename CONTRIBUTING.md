# 参与开发

## 跑起来

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
python tests/test_primolix_e2e.py                # 端到端：建/查/增量/删除/重载
python tests/test_kernel_unit.py                 # 内核不变量：往返/墓碑/追加/公式/兼容
python tests/test_staged_publish.py              # 段式发布：挂载/折叠/卸载与逐位回滚（T1–T7）
python examples/run_examples.py                  # 可跑示例（自带语料，幂等）
python -m compileall -q primolix                 # 语法体检
```

三个测试脚本都不需要外部数据，逐项打印 `PASS`/`FAIL`，有失败就返回非零 —— CI 直接跑它们：
`tests/test_kernel_unit.py`（**23 项**，U1–U23）、`tests/test_staged_publish.py`（**7 项**，T1–T7）、
`tests/test_primolix_e2e.py`（**10 项**，1a/1b/2–9）。

## 改代码时的七条红线

1. **不许静默。** 失败要么抛异常、要么显式返回状态。
   例：`update()` 必须返回 `{'kind': 'noop'|'incremental'|'rebuild'}`；词表外词不许静默丢掉。
2. **同一套算式要逐位一致。** "应当等价"的路径（换参 / 增量 / 重建 / 折叠）比较时要求 `max|Δ| == 0`。
   只有不同实现或不同累加序才用相对容差，而且要同时报出绝对值。
   反过来做（对独立实现要求逐位、或对同一实现放宽到容差）两边都会误导。
3. **每条契约配一条测试。** 新增或修改对外行为，就在 `tests/` 加一条能钉住它的检查
   （例：`U9` 钉住"旧格式串仍可加载"）。
4. **对外主张要带边界。** 文档里每个能力词后面都要写清适用面；
   未实现的东西不许写进功能表 —— 如果只是设计上可行，就写明"设计上可行，尚未实现"。
5. **改名要审计所有"被写进盘或协议"的名字**：格式串、环境变量、段名前缀、默认目录名、类名 ——
   每一个都要留旧名回退（历史教训：包改名漏了 `fmt`，会导致旧索引全部加载不了）。
6. **别手工删索引里的任何文件**（必需性见 `docs/index_format.md`）；要清理就整目录删掉重建。
7. **写含中文的 f-string 时，引号一律用「」**（ASCII 双引号会提前闭合字符串，变成语法错，
   而报错信息通常看不出真正原因）。历史教训：这个仓库已累计出现 5 次同类问题 —— 写完记得跑一次
   `python -m compileall -q primolix`，它是这条的机械防护。

## 测试怎么写

- 三个入口：`tests/test_primolix_e2e.py`（端到端，10 项）、`tests/test_kernel_unit.py`（内核不变量，
  23 项 U1–U23）、`tests/test_staged_publish.py`（段式发布，7 项 T1–T7）。
- **参数形态别记错**：内核是 `score_all(qtoks)`（**token 列表**），索引层是 `query(text)`（**文本**）。
  给内核传字符串会被当成字符集合，静默全 0 分（历史教训：测试里出现过一次）。
- **对照组的状态要与被测组一致**（写测试时也要做"活跃集对齐"）。

## 提交

- 一次提交一件事；信息写清「改了什么 / 为什么 / 验证了什么」。
- 带读数的改动，把读数或复现命令写进提交信息。

## 许可

提交即表示同意以 **MIT** 许可发布本仓库的内容。
