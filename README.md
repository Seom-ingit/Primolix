# primolix

一个稀疏检索内核：索引里只存可逆的原语（tf、df、dl 和墓碑），分数在查询时算出来。

*English docs: [README.en.md](README.en.md)*

## 安装

```bash
pip install -e .          # 现在：从源码装（PyPI 发布后用下面那条）
pip install primolix      # 发布到 PyPI 之后
```

Python >= 3.9。依赖：`numpy`、`scipy`（**缺 scipy 会直接报错，不会降级**）。

**中文分词用 `jieba`**：未安装时自动降级为"字 + 相邻二元"，仍可用但精度较低 ——
降级状态可查：`primolix.kernel.tokenizer_status()`。可选依赖：`pip install primolix[dense]`（稠密精排，缺了自动退化为纯 BM25）、
`psutil`（并行分词的内存感知 worker 数，缺了只看 CPU 核数）、`rich`（命令行彩色输出，缺了降级为纯文本）。

## 十秒演示

```bash
python -m primolix index  examples/corpus/ --out examples/corpus/.idx   # 建索引
python -m primolix query  "真增量 原语" --out examples/corpus/.idx --k 3
python -m primolix update examples/corpus/ --out examples/corpus/.idx   # 真增量（改一篇只做那一篇）
```

真实输出（本机，语料是仓库里 `examples/corpus/` 的十篇文档）：

```
$ python -m primolix index examples/corpus/ --out examples/corpus/.idx
✔ 建索引完成

$ python -m primolix query "真增量 原语" --out examples/corpus/.idx --k 3
───────────────────────────  top-3 · BM25 · 569ms  ────────────────────────────
  # 1 3.653 01_incremental.md:1
     真增量写入：改一篇文档只做那一篇的工作，不重建整库。墓碑标记旧行，新内容追加到尾部。
  # 2 1.962 02_primitive.md:1
     原语层：索引里只存 tf、df、dl 与墓碑。分数（BM25 权重）是这些原语的函数，查询时算出来。

# 改一篇，复用已有词 → 走增量
$ python -m primolix update examples/corpus/ --out examples/corpus/.idx
✔ 真增量更新完成 (tomb=1 append=0)

# 改一篇，引入新词 → 按契约回退重建（不静默丢词）
$ python -m primolix update examples/corpus/ --out examples/corpus/.idx
✔ 整库回退重建完成 (原因: oov=6)
```

两条 `update` 的差别就是"词表内 / 表外"，契约见 [docs/api.md](docs/api.md) 的契约表。

## 该不该用

取舍与适用边界写在 **[docs/when_to_use.md](docs/when_to_use.md)**（包含"别选它"的场合，与本文档同样直接）。
一句话：**改动频繁、要调参、要解释、要审计** 就合适；**只读且查询密集、延迟极紧** 就用预计算方案。
接口清单（含契约与坑）：**[docs/api.md](docs/api.md)**（[EN](docs/api.en.md)）；索引文件与兼容：**[docs/index_format.md](docs/index_format.md)**（[EN](docs/index_format.en.md)）· 选型指南英文版 [docs/when_to_use.en.md](docs/when_to_use.en.md)。

### 分布式部署（精确分层）

- **跨机、多机**：本包**没有**分片路由，也**没有**协调层。这是"未实现"，不是机制劣势。
- **单机之内，包里已经有这几件**（都能独立使用）：
  - **段式多片**（**段 = 挂在主表旁的只读分块，不动主表**）：`mount_segment` / `fold_segment` /
    `unmount_segment` 把只读段挂进同一份索引（含卸载），与全库重建逐位一致（测试 U10–U12）；
  - **跨进程共享 CSC**：`TC_SHARED` 让多个进程共用同一份分块 CSC（默认关；开与不开的条件见
    下文的「性能开关」一节）；
  - **分片建库的两个关键小件**：`SparseBM25(..., vocab=...)` 锁定列空间（出现表外词**拒绝静默扩表**）、
    `stream="ids"` 只做一遍分词后按 token id 建库。逐片 append 的**编排**在源工作区的 `research/`
    台子里（`shard_stream_build_probe.py` / `shard_sim_probe.py`，两个台子均未随本发布包提供），
    **尚未成为包的 API**。
- 低常驻私有内存确实有利（一台机器能放更多分片或副本；索引段只读、天然可复制），
  但**"低内存换每核吞吐更高"并不成立**：查询在 CPU 上现算分数，低内存买到的是**部署密度**，
  不是每核吞吐。
- 分片真正的难点是全局 `idf`（依赖全局 `df`），而这恰好是"把 `df` 当原语存"能精确解决的：
  汇总全局 `df` 就得到精确 idf，**不必重建每个分片**。
- 层次血缘索引（`primolix.lineage`）未随本发布包提供（尚无回归测试）。

## 名字

primolix = primo / primitive（原语）+ LIX。

LIX 取自上一代的 ZELIX（Zero-Encoding Lazy Indexing & eXecution，零编码懒索引与执行）。
两代的关系是：

- **ZELIX（上一代，姊妹项目）**：零编码索引 + 查询时懒编码候选池重排，主张在 8.84M 篇规模上
  "DPR 级质量、BM25 级建库成本"。
- **primolix（这一代）**：从 ZELIX 的内核里抽出原语层（tf / df / dl + 墓碑），把重点放在
  索引生命周期上：增量写入、换公式不重建、归因、对账、多视图。

ZELIX 是上一代的内部代号：本包不以它命名任何模块、类或导出。上面提到它，只是为了解释 LIX 的来源。

名字只承诺这个包里真有的东西：原语，以及惰性索引与执行。它不是多模态系统，只处理文本
token。它与做 Java 混淆器 KlassMaster 的 Zelix Pty Ltd 无关，只是共享 LIX 这三个字母。

## 这是什么项目

primolix 是一个 BM25 稀疏检索内核，Python 实现，纯 CPU 运行。

它与多数检索库的区别在于怎么处理"分数"。常见做法是把算好的加权分数当作索引的一部分
写进磁盘：于是改一篇文档要重建整库，调一个参数也要重建，想知道"为什么这篇排前面"只能拿到
一个总分，想确认"改完还对"只能重建之后再对账。

primolix 把 tf、df、dl 这些原语存下来，分数是它们的函数。同一份索引因此可以直接做四件事：

- 就地改文档，只做改动的部分；
- 改 k1 / b / idf，不重建、不动物料；
- 把一篇文档的分数按词拆开，并回答"如果改参数，这篇会变成几分"；
- 改动之后只核对原语，成本与改动量成正比。

代价是**默认**在查询时现算分数：在 44,757 篇文档的小索引上是 **0.049 ms/查询**（见下文数据表）。
从本版起默认路径带两层可选优化，且都与旧路径**逐位**一致：

**查询性能（默认开启）**：50 万篇文档规模、同一批 100 条查询、同批读数 ——
旧默认 **5.38 ms/查询** → **新默认 2.70 ms**（1.99 倍）→ 再开**可选物化层 `W_CACHE`** **1.16 ms**（4.65 倍）；
同批对照（预加权方案）**1.39 ms**。三档全程 `max|Δ| = 0`。
三个开关 `DN_PRECOMP`、`ADD_AT`、`W_CACHE` 由 `FAST_SCORE` 总闸控制，细节见 [docs/api.md](docs/api.md)。

**边界**：物化层默认只在内存（`nnz×4 B`，100 万篇文档约 165 MB）；若要长期用，可以在**建库时写进磁盘**
（`--w-cache`），装载时按内容型键复用。它依赖命中（换 `k1`/`b`/`avgdl` 或追加墓碑会让它失效，
之后自动回退现算）；本机跨批次绝对值波动 20%–40%（个别批次可达 2 倍），所以上面的倍数只在同一批次内成立。

适合的场景：文档经常增删改、需要反复调参、需要解释排序结果、需要审计检索过程。
不适合的场景：只读且永不改动且不开物化层、把单次查询延迟压到极限。

## 项目特点

1. **真增量写入（词表内）。** 改 100 篇文档只做这 100 篇的工作，其余 44,657 篇不碰。
   边界：改动里出现**词表外的新词**时，默认走**结构级重建**（`O(V+nnz)`，本机约 4–6 s），
   并返回 `kind='rebuild'`（不会静默丢词）。另一条路线**已接进内核（最小档，三阶段全 API）**：
   `SparseBM25.mount_segment(...)` 把新词放进**只读段**接住（段带原语、打分用**合并 `df`**、
   词表代次可查），**与全库重建逐位一致**（实测 `max|Δ| = 0`），**不动主表 `T`**、不写盘；
   `fold_segment(tag)` 再把它并进主表（折叠前后分数不变），`unmount_segment(tag)` 可撤回
   （**后进先出**，且仅对未折叠段）。**并发换段已实测不是原子的**：无锁时约 **3%**（多次实测 2.3%–3.7%）的并发读数
   撕裂、其中 **96% 是静默错分**，所以**内核默认已加可重入读写锁**（`SWAP_LOCK=True`：读者并发、
   写者独占，实测 **0 撕裂**；代价约 **+3.4 µs/查询**，单线程可关）；门面层再加一层更粗的锁，
   让"墓碑＋追加＋写入磁盘"这类**跨调用组合**也原子。详见 [docs/api.md](docs/api.md) 的契约表与第 8 节。
2. **换公式不重建。** 改 k1 / b 是两次属性赋值；改 idf 是重算一个长度 V 的向量；物料
   一个字节不动。
3. **回滚与多视图。** 视图是 `(段清单, 墓碑, 参数, 词表代次)` 四元组，切视图只换清单，
   物料不动，因此可以同时开多个视图（版本对比、A/B 调参）。**边界**：视图 API 随包提供
   （见 [docs/api.md](docs/api.md) 的视图族），但**发布件的 CI 未覆盖视图路径** ——
   e2e 与内核单元测试覆盖的是建库、查询、真增量、换公式、删除、重载，以及段的挂载/折叠/卸载（U10–U12）。
4. **可归因。** 任意 (查询, 文档) 的分数可以按词拆成 `idf × tf 饱和 × 文档长度` 的贡献，
   并能当场算出换参数后的分数。
5. **增量对账。** 改动之后只需核对原语即可确认结果，不必重建整库。
6. **压实。** 长期只追加会让索引分块数增长，查询随之变慢；压实把小块并回一大块，恢复
   查询速度。

## 项目数据

测试环境：同一台 Windows 机器、同一份索引（44,757 篇文档，词表 73,849，nnz 2,618,712，
磁盘 26.9 MB），30 条查询。每个数字都在独立进程里跑过多轮，取稳态中位数。

| 项目 | 数值 | 条件 |
|---|---|---|
| 增量写入 | 0.22 µs / token | K=100 批，稳态 |
| 换 k1 / b | 1.2 µs | 两次属性赋值 |
| 换 idf | 5.3 ms | 重算长度为 V 的向量 |
| 全量重建（对照） | 5.7 s | 同样的语料 |
| 切视图 | 10.9 ms | 7 个段，不含初次载入 |
| 归因反事实 | 0.9 ms + 1.8 ms | 重算 idf + 30 条查询 |
| 增量对账 | 8.7 ms | 改 100 篇 + 追加 100 篇 |
| 查询 | 0.049 ms / query | 快速路径默认已开；关闭时 0.132 |
| 常驻内存（私有） | 2.80 MB | 不含共享页 |
| 冷启 | 36 ms | 加载索引 |
| 压实 | 21 ms | 17 块并成 1 块，查询从 0.217 恢复到 0.065 ms/query |

正确性（每项都是与"整库重建"或"从头建库"的对账结果）：

- 增量写入后的分数与重建逐位相同；
- 换 k1 / b 或 idf 后的分数，与用同样参数从头建库逐位相同；
- 归因的逐词贡献之和与打分函数输出逐位相同；
- 对账时把 df 改坏一项能被发现。

口径说明：同一批次内部的比较可信；跨批次的绝对值会有 20%–40% 的波动（个别批次可达 2 倍），
所以上表是本机、本批的读数，不是跨环境的保证。不开物化层时查询慢于预计算方案（开启 `W_CACHE`
且索引只读时可以反超）、常驻内存含共享页时数字会变大，这两条在数据里都能看到，不另行辩护。

## 性能开关：三个默认开，两个由你决定

默认路径已经把"白拿"的优化全打开了，它们**逐位**不改变结果，你不需要为它们做任何事：

| 开关 | 默认 | 作用 | 代价 |
|---|---|---|---|
| `FAST_SCORE` | 开 | 查询快路径（总闸） | 无（逐位；关掉只会更慢） |
| `DN_PRECOMP` | 开 | 预计算每行规范化项 | N 级 4 MB（100 万篇文档）＋ 首次约 4 ms |
| `ADD_AT` | 开 | 无缓冲累加 | 无（逐位、零内存） |
| `SWAP_LOCK` | 开 | **并发安全**：换段与查询互斥（读者并发、写者独占） | 每查询约 **+3.4 µs**（小索引 17.6%，4.5 万篇约 7%） |

下面两个**默认关**，不是遗漏，而是它们各拿一根轴换延迟，得由你的场景决定。
**`W_CACHE`（物化层）—— 用内存换查询延迟。** 命中时每个 posting 只做一次加法。
- 值得开：语料基本只读、参数不再调、常驻内存宽裕，且每进程会跑很多查询
  （按 500k 读数估算，约 **200 次查询**才回本；此为估算，未直接实测）。
- 代价：常驻 `nnz×4 B`（100 万篇文档约 **165 MB**；884 万篇文档按外推约 **1.465 GB**）；首次命中要建造
  （100 万篇文档约 **0.5–0.6 s**）；**换 `k1`/`b`/`avgdl`，或写入与墓碑改变块结构后自动失效**，回到现算。
- 长期使用走**建库时写入磁盘**：命令行 `--w-cache`（Python 侧 `save(..., w_cache=True)`），之后装载即用 ——
  100 万篇文档实测首查 **238.3 ms 降到 3.1 ms**，代价是盘上多 `nnz×4 B`。注意同一目录就地重存会撞 Windows 的
  mmap 锁，请写到新目录。
- 不开也不会错：未命中自动回退现算。

**`TC_SHARED`（多进程共享那份 CSC）—— 用部署形态换常驻内存。**
- 值得开：**同一份索引被两个以上进程长期共享**（worker 池、多租户）。
- 单进程不要开：只多付一次发布。
- 前提与代价：**写入会让整份共享键失效**（键是 `路径|mtime|shape|nnz`），其它进程需要重新发布；
  **跨操作系统语义尚未验证**；报内存必须**两口径**（`private` 会明显变小，但 `RSS` 会变大，因为共享页计入 RSS）。
- 打不开会**安全回落**到私有 CSC：不会算错，只会慢。

**不建议开**：`WRITE_DICT`（实测无收益）、`MATVEC`（实测反而慢 27%）。

更细的开关语义、失效条件与回本口径见 **[docs/api.md](docs/api.md)** 的"开关怎么选"与第 5 节契约。

## 用法

想先跑起来看：**`python examples/run_examples.py`**（自带 10 个语料文件，演示建索引、查询、
真增量、换公式、归因，可重复跑）。

**门面（推荐入口）**：一句话一个动作；**并发安全默认开**（读者并发、写者独占）：

```python
import primolix

idx = primolix.build("my_docs", "my_docs/.idx")   # 建索引（返回门面对象）
idx = primolix.open("my_docs/.idx")               # 打开已有索引（不存在就报错，不猜）
idx.search("检索 内核", k=10)                      # 文本进、结果出（自动分词）
idx.add("my_docs")                                # 增量（词表外则如实回退重建）
idx.tune(k1=2.0, b=0.3)                           # 换公式：不重建、不动物料
idx.explain("检索 内核")                           # 归因：分数按词拆开
idx.stats(); idx.segments(); idx.views(); idx.compact()   # 概览 / 段 / 视图 / 压实
```

命令行（`python -m primolix --help` 看全部）：

```bash
python -m primolix info    --out my_docs/.idx                 # 概览：规模 / 开关 / 段 / 视图
python -m primolix params  --out my_docs/.idx --k1 2.0        # 换公式（--save 写回索引）
python -m primolix config  --set W_CACHE=true                 # 开关（进程级，不写入磁盘）
python -m primolix explain "检索 内核" --out my_docs/.idx      # 归因：分数按词拆开
python -m primolix segment list --out my_docs/.idx            # 段：list / mount / unmount / fold
python -m primolix view    list --out my_docs/.idx            # 视图：list / attach / detach / fuse
python -m primolix compact --out my_docs/.idx                 # 压实：把已挂段并进主表
```

命令行：

```bash
python -m primolix index  my_docs/  --out my_docs/.idx   # 建索引
python -m primolix query  "检索 内核"  --out my_docs/.idx --k 10
python -m primolix update my_docs/  --out my_docs/.idx   # 真增量：只处理改动过的文件
python -m primolix browse --out my_docs/.idx --files
```

Python：

```python
from primolix import Primolix

z = Primolix.build("my_docs", "my_docs/.idx")
z = Primolix("my_docs/.idx")
scores = z.query("检索 内核", k=10)
```

直接用内核：

```python
from primolix.kernel import SparseBM25

bm = SparseBM25(texts, lambda s: s.split(), k1=1.5, b=0.75)
bm.update_docs([3, 7], [["新", "内容"], ["另一", "篇"]])   # 就地改
bm.k1, bm.b = 2.0, 0.3                                     # 换公式，不重建
```

关于输出：库的进度信息都写到 **stderr**（不会污染调用方的 stdout）；需要完全安静时设
`primolix.core.VERBOSE = False`。命令行入口不受影响，结果照常打到 stdout。

## 测试

```bash
python tests/test_primolix_e2e.py     # 端到端：建索引、查询、真增量、删除、重载
python tests/test_kernel_unit.py      # 内核不变量：往返、墓碑、追加、公式、段三阶段、开关逐位
python tests/test_staged_publish.py   # 段式发布：挂载 / 折叠 / 卸载与逐位回滚（T1–T7）
```

三个脚本各自逐项打印 `PASS` / `FAIL`，有失败就返回非零；不需要外部数据。
覆盖范围：内核 **U1–U23**（含段的挂载 / 折叠 / 卸载 U10–U12；U20 钉"重载后可继续更新"与
"`.mm/` 世代键不符则拒用"）＋ 索引层 10 项端到端检查 ＋ 段式发布 7 项（T1–T7）。
**未覆盖**：视图族。

## 目录

```
primolix/    包：命令行、索引对象、BM25 内核、一句话门面、词表载体、跨进程共享
             （cli.py · core.py · kernel.py · api.py · tc_shared.py · vocab_mmap.py）
tests/       端到端与内核单元测试（英文输出）
examples/    可跑示例：run_examples.py（内核与索引对象）
docs/        接口与索引格式说明、选型取舍，以及研究记录
research/    台子与读数的出处索引：上表每个数字来自哪个台子在该目录 README 里列出（另含分片建库与
             多进程台子）；台子脚本与读数文件未随本发布包提供
```

## 许可

MIT，见 LICENSE。参与开发见 [CONTRIBUTING.md](CONTRIBUTING.md)；版本与兼容承诺见 [CHANGELOG.md](CHANGELOG.md)。
