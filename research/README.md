# research/ — probes and readings

*中文说明在下方。Short English note at the bottom.*

## 这一目录是什么

这里是**方法学附档**：记录读数的出处（哪个台子产出哪个数字）与当时的测量口径。
**台子脚本（`*.py`）与原始读数（`*.json`）均未随本发布包提供**，本发布包的 `research/` 里只有这份说明。
项目 README 的"项目数据"表、`CHANGELOG` 与 `docs/` 里每个数字来自哪个台子，在本文件里列出。

## 先说清边界：这不是"一键可复现"

源工作区里的台子读的是**未随本仓库发布的私有语料**，并且直接写死了路径。源工作区的静态清点（92 个 `.py`）：

| 项（源工作区静态清点） | 数量 |
|---|---|
| 台子总数 | 92 |
| 直接写死绝对路径（如 `D:\rag_datasets\...`、`D:\question\rag像素化优化\...`） | **30** |
| 需要真实语料或查询集文件 | **37** |

（两者有重叠。）所以：**拿到本仓库不等于能复算出 README 里的数字**。README 的"项目数据"表也标了这一点
（同批次内可比、跨批次绝对值会漂 20%–40%，个别批次可达 2 倍）。

## 怎么用这一目录

1. **当方法与口径的参考。** 这里沉淀的可复用做法比单个数字更保值，例如：
   - 读数分**三个时刻**（冷 / 暖 / 稳态），报哪个要写清；
   - 常驻内存报**两个口径**（`private` 与 `RSS`），因为共享页会让两者给出相反结论；
   - 延迟用**交错 A/B**（同进程先后跑会系统性慢化 ≈1.9×）；
   - 每次同批对照带**查询集指纹**；
   - 吃内存的台子外面套一层护栏（进程内自报 RSS、kill 线、读数增量写入磁盘）。
2. **在你自己准备的同格式语料上重跑**（需自备台子 —— 它们未随本发布包提供）。台子按 `primolix.*`
   导入（并把仓库根放进 `sys.path`），所以只要换掉数据路径，就能在等价规模上复算。语料格式：

   - 文档语料：**JSONL，一行一条**，每行一个对象，正文字段是 `text`
     （`{"text": "..."}`，可带其他字段）；
   - 查询集：**JSONL，一行一条**，查询字段是 `q`（`{"q": "..."}`）。

3. **注意台子会写临时产物**（`.dbg/`、`.dbg_guard/`、`_tmp_*.npz` 等）。这些都已进 `.gitignore`。

## 源工作区里已知的不可直接运行项（如实列出）

- `latency_ab.py` 里有两路**上一代内核的对照臂**（标签 `zelix_*`）—— 那需要旧内核与旧索引，
  本仓库里没有，这几路跑不起来，只作口径参考（该台子未随本发布包提供）。
- 少数台子把"工作区布局"写进了逻辑（例如扫描时的排除目录名）。在发布仓里目录名不同（这里是 `research/`），
  所以这类台子即使能跑，**扫描口径与当时的读数不完全一致**，只当方法参考，不要当同批对照。
- `kernel_patch/` 是当时的内核补丁与冒烟脚本，在源工作区里保留作证据（未随本发布包提供），
  不保证在当前版本上仍然适用。

## English summary

This release ships **this note only**: the **probe scripts and raw readings** behind the numbers in the
project README, CHANGELOG and docs are **not shipped with this release**. It is a methodology annex,
**not a one-click reproduction**: in the source research tree, 30 of the 92 scripts hard-code absolute
paths to private corpora, and 37 need real corpus/query files. If you have the probes, supply JSONL files
of the same shape (one object per line: `{"text": ...}` for documents, `{"q": ...}` for queries) and they
will run against your data. The reusable part is the measurement discipline (three time points,
two memory calibers, interleaved A/B, frozen query-set fingerprints, memory guard rails).
