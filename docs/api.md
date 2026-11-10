# API 参考（primolix）

*English: [api.en.md](api.en.md)*

> 只列公开接口与契约；私有方法不写。文中标「契约」的几条都有对应测试钉住，
> 已知的坑也一并列出。索引文件与兼容性见 [`index_format.md`](index_format.md)。

## 1. 顶层

```python
from primolix import Primolix, SparseBM25, zh_tokenize, simple_tokenize, log
```

| 名字 | 是什么 |
|---|---|
| `Primolix` | **索引对象**：面向"一个目录的文档"，负责扫描/分块/持久化/视图 |
| `SparseBM25` | **内核**：BM25 稀疏矩阵 + 真增量 + 分块 CSC 查询（`primolix.kernel`） |
| `zh_tokenize` / `simple_tokenize` | 中文（jieba，缺则降级为字+相邻二元）/ 英文分词 |
| `log` | 日志钩子（默认写 **stderr**；库路径下受 `core.VERBOSE` 控制） |

## 2. `Primolix`（索引对象）

```python
z = Primolix.build("docs/", "docs/.idx")      # 建；classmethod，返回实例
z = Primolix("docs/.idx")                      # 打开已有索引
```

| 方法 | 说明 |
|---|---|
| `Primolix.build(root, out='.primolix_index', dense=False, model_path='', workers=0, skip_dirs=None)` | 建索引（classmethod）。`dense=True` 时另落 `dense.npy` |
| `Primolix(index_dir, dense=False, model_path='')` | 打开索引（读盘） |
| `query(q, k=10, dense=False)` | 检索，返回 `[{'rank','score','file','line','text'}]`，**按分数降序**，遇 `score<=0` 截断 |
| `update(root, workers=0, verify='auto')` | 按源文件差异做增量，返回 `{'kind', 'detail', 'verify'}`（见第 5 节契约）。`verify` 三档：`'auto'`（有 `(size, mtime_ns)` 印记就走 `'stat'`）/ `'stat'`（只比元数据，不读内容）/ `'full'`（每个文件算 md5） |
| `rebuild_in_place(root, workers=0)` | 整库重建到同一目录（词表外回退、或显式重建） |
| `info` | 属性：`N`（chunk 数）、`V`、`size_mb` 等 |
| `browse(file=None, page=0, n=20)` | 浏览分块（CLI `browse` 用它） |
| `list_files()` | 列已入库的文件 |

**视图族**（多版本、A-B、稠密精排候选）：

| 方法 | 说明 |
|---|---|
| `attach_view(name, vectors, meta_positions=None, *, ...)` | 挂一个视图（向量与全局 docid 对齐） |
| `detach_view(name)` / `view_report()` / `covered_mask(view_name)` | 卸载 / 报告 / 覆盖掩码 |
| `save_view(name, docids, vectors, meta=None)` / `load_views(extra_dirs=None)` | 视图写入磁盘 / 载入 |
| `fused_scores(qtoks, qvec, view_name, alpha=0.35)` | 稀疏＋稠密融合打分 |

**边界**：视图族随包提供，但**发布件的 CI 未覆盖视图路径**（e2e 与内核单元测试覆盖建库、查询、
真增量、换公式、删除、重载，以及段的挂载/折叠/卸载 U10–U12），用前请按本节接口自行验证。

## 3. `SparseBM25`（内核）

```python
from primolix.kernel import SparseBM25

bm = SparseBM25(texts, lambda s: s.split(), k1=1.5, b=0.75)   # 从文本建
bm = SparseBM25.load("idx/bm25.npz")                          # 从盘加载（classmethod）
```

| 成员 | 说明 |
|---|---|
| `score_all(qtoks)` | **只收 token 列表**（不是字符串），返回 `float32[N]` 分数 |
| `update_docs(rows, toks_list)` | 真增量 upsert：**墓碑旧行 ＋ 尾部追加新行**，返回新行号 |
| `append_docs(toks_list)` | 只追加（不做墓碑） |
| `oov_terms(toks_list)` | **词表外词预检**（空集表示可安全走真增量） |
| `live_rows()` | 当前存活行号（无墓碑时等于 `range(N)`） |
| `save(path, w_cache=False)` / `load(path, N=0, V=0)` | 写入磁盘 / 加载（`load` 兼容 `zelix-bm25-v2/v3` 旧格式串）。`w_cache=True` 时**额外**把物化层 `w` 写进 `.mm/`（`w_*.npy` ＋ 内容型键 `w_key.txt`；装载时键不符**拒用**），冷启动因此不必重建（1M 实测首查 238.3 → 3.1 ms；盘 `+nnz×4 B`） |
| `SparseBM25(docs, tok, k1=1.5, b=0.75, workers=0, stream=False, vocab=None)` | 构造函数。**`vocab=`**＝给定**列序词表**（分片建库要求列空间一致；出现表外词**拒绝静默扩表**并报错）· **`stream=True`**＝两遍各自分词、不驻留 `doc_toks`（内存 `O(N×len)` → `O(三元组)`）· **`stream="ids"`**＝按 token id 建库（配合 `vocab=` 做逐片建库） |
| `mount_segment(seg_dir, tag, new_terms, dl, rows=None, *, generation=None)` | **挂一段只读 postings**（三个 `.npy`，CSC，列号落在"主表加宽后"的空间里）：把段当分块挂进查询路径，**合并 df**（老词加、新词扩列），扩词表/`dl`/`N`，重算 `_idf`，结果**与全库重建逐位一致**（实测 `max\|Δ\| = 0`）。不写盘，不改主表 `T`。代价：词表用普通 dict 覆盖（载体只读） |
| `unmount_segment(tag)` | **撤回一个已挂的段**：把 `mount_segment` 做过的事**逐项撤回**（摘块、减回 `df`、缩词表/`dl`/`dead`/`N`、重算统计），分数**逐位回到挂载前**（测试 U12）。两条前提都会显式报错：① 该段**未被折叠**（折叠后内容已在主表，只能重建）② **后进先出**（必须是最后挂的那个，否则行/列账会算错） |
| `fold_segment(tag)` | **把已挂的段折叠进主表**（扩列 ＋ 追加行，一次 `O(nnz)`），折叠后查询只走主表。`df`/`dl`/词表在挂段时已合并，所以本方法**不再动它们**（否则双重计数）。返回 `{rows, ms, nnz_before, nnz_after, ...}`，并把它移入 `folded_report()` |
| `fold_all()` | **把全部已挂段一次性折进主表**（按基行号排序，各块补齐到同一列宽后追加，最后断言主表行数等于 `N`）。单段折叠的方法在多段下不成立（它断言"折完主表行数等于 `N`"，而 `N` 含**所有**已挂段），所以多段请用本方法；门面的 `compact()` 会自动选用。返回 `{tags, rows, ms, nnz_before, nnz_after, N, V}` |
| `segment_report()` / `folded_report()` | 已挂段清单（含 `gen` **词表代次**、`row_base`、`m_new`、`df_added`、`bytes`）与已**折叠**段的留档 |
| `topk(qtoks, k=10, *, sort=True)` | **打分与取 top-k 一次做完**，返回 `(行号, 分数)`（默认按分数降序）。比"`score_all` 后再 `np.argpartition(-s, k-1)`"**省掉对整条分数向量取负的那次拷贝**（500k 上实测省 **0.47 ms / 2 MB**）。与 `score_all` 一样**只收 token 列表**；k 边界**并列**时取法不定（**分数值一定相同**） |
| `k1` · `b` · `avgdl` · `dl` · `df` · `N` · `V` · `nnz` | 参数与统计量（`V`/`nnz` 是 property） |

**类属性（开关）**：

| 开关 | 默认 | 作用 |
|---|---|---|
| `SparseBM25.T_SLACK` | `1.5` | 行方向容量预分配倍率（append 摊销 `O(新 nnz)`） |
| `SparseBM25.FAST_SCORE` | `True` | **查询快路径**（省 `getcol` 的对象与拷贝，去掉中间变量），**逐位**（只做不改变浮点运算顺序的改动）。下面三个开关只在它为 `True` 时生效 |
| `SparseBM25.TC_SHARED` | `False` | 把分块 CSC 放进共享内存（多进程共用一份） |
| `SparseBM25.WRITE_DICT` | `False` | 写路径改用 dict 查词（实测**无收益 0.99×**，不建议开） |
| `SparseBM25.DN_PRECOMP` | `True` | 预计算"每行规范化项" `dn`（**N 级**，1M 约 4 MB），**+8–12%**，**逐位**（测试 U14）。只在 `FAST_SCORE=True` 时生效 |
| `SparseBM25.ADD_AT` | `True` | 累加改用 `np.add.at`（**无缓冲**），**+40–65%**（500k 命中路径 1.87→1.02 ms），**逐位**（测试 U17）。收益依赖 numpy 实现（**可能回退，不会算错**）。只在 `FAST_SCORE=True` 时生效 |
| `SparseBM25.W_CACHE` | `False` | **物化层 `w`**（`nnz×4 B`，1M 约 165 MB）：查询时**每 posting 一次加法**，命中 **+60%**（合计 **2.4–3.2×**），**逐位**（测试 U16）。换参**自动失效**（键含 `k1/b/avgdl/代次/块数`）；首次命中付 **O(nnz) 建造**（1M 约 0.6 s）；**可选写入磁盘**（`save(w_cache=True)` 写 `.mm/w_*.npy` ＋ 内容型键 `w_key.txt`，装载时键不符**拒用**）；**段在场时不参与**（批次内查询走现算，折叠后的第一次查询重建一次） |
| `SparseBM25.MATVEC` | `False` | 手写矩阵化，**实测负结果**（500k **0.728×**，反而慢 27%），保留为留档，不建议开 |
| `SparseBM25.SWAP_LOCK` | **`True`** | **并发安全：换段与查询之间用可重入读写锁互斥**（读者并发 · 写者独占 · 写者优先）。动因：无锁并发换段实测 **约 3% 的读数撕裂**（多条读数 2.3%–3.7%），其中 **96% 是静默错分**；加锁后实测 **0 撕裂**。代价 **+3.43 µs/查询**（小索引 17.6%，44,757 篇约 7%）；**单线程或没有并发换段**时可设 `False` 拿回 |

**开关怎么选（代价与回本）**：

- `FAST_SCORE` / `DN_PRECOMP` / `ADD_AT`：**保持默认开**。三者都逐位（测试 U14/U17 钉住）、不新增常驻
  （`dn` 是 N 级，100 万篇文档约 4 MB）、首次成本是 `O(N)` 级（约 4 ms），关掉只会更慢。
- `W_CACHE`：**用内存换延迟**。常驻 `nnz×4 B`（100 万篇文档约 165 MB；884 万篇文档按外推约 1.465 GB）；
  首次命中建造 `O(nnz)`（100 万篇文档约 0.5–0.6 s）；换 `k1`/`b`/`avgdl`，或写入与墓碑改变块结构后
  **键变即失效**，自动回到现算。适合"语料只读、常驻宽裕、每进程查询多"；查询少时是净亏
  （按 500k 读数估算，约 200 次查询回本）。长期使用请走**索引期写入磁盘**（`save(..., w_cache=True)`，
  命令行 `--w-cache`），装载后按内容型键 `_w_persist_key()` 判断能否复用（100 万篇文档实测首查 238.3 ms 降到 3.1 ms）。
- `TC_SHARED`：**用部署形态换常驻**。收益要求"同一份索引 ＋ 多个长期进程"；**单进程不要开**
  （只多付一次发布）。写入会让 `_tc_key()`（`路径|mtime|shape|nnz`，即**整份索引**）失效，
  其它进程须重新发布；**跨操作系统语义未验证**；内存必须**两口径**报（`private` 下降而 `RSS` 上升，
  共享页计入 RSS）。不可用时自动回落私有 CSC，分数不变。
- `WRITE_DICT` / `MATVEC`：**不建议开**（前者实测无收益，后者实测慢 27%）。

## 4. 模块级工具（`primolix.kernel`）

`tokenizer_status()`（查分词是否降级）· `zh_tokenize` / `simple_tokenize` · `auto_workers()`（内存感知的并行度）·
`parallel_tokenize` / `parallel_tokenize_stream` · `log` · `full_hash` · `ViewError` · `PreallocCSR`（容量预分配 CSR）。

`primolix.core` 还导出可调常量：`MAX_CHUNK`（默认 1200）· `CODE_BLOCK_LINES`（80）· `MAX_FILE_BYTES`（2 MB）·
`EXCLUDE_DIRS` / `EXCLUDE_EXT` / `CODE_EXT` / `PROSE_EXT`（扫描与分块规则）。

**选择段（0.2.0）**：`primolix.core.TOPK_APPROX`（默认 `True`）· `TOPK_FRAC`（32）· `TOPK_SHRINK`（0.98）·
`topk_stats(reset=False)`。`topk_stats()` 返回本线程 `_topk_idx` 的 `calls / approx / exact / fallback / cand`
—— 用来**自证"这一臂真的走了哪条路"**（近似路径没被触发时，"两臂打平"是没有信息量的）。
`TOPK_APPROX = False` 即完全确定（只有精确路径）。

**`primolix.span`（meta 层定位，不进打分链路）**：`hit_span(text, query)` 给最小覆盖跨度 `(start, end)`
（整串命中优先，无词法命中返回 `None`）· `hit_ranges(text, tokens)` 给每个词的全部出现区间 · 
`query_tokens(q)` 与前端 `lib/span.ts` 同款切分（空白 + 中英标点，不做真分词）。口径是**纯子串匹配**，
所以调用方与前端可以逐行对拍；它**不参与 BM25 打分**，也**不改 `tokenize` 的工作**。

## 5. 契约与坑（实测）

| # | 契约 / 坑 |
|---|---|
| 1 | **`update()` 的返回值是契约**：`{'kind': 'noop'\|'incremental'\|'rebuild', 'detail': str}`。**词表外词返回 `kind='rebuild'`**（整库回退重建，不静默丢词）；词表内改动返回 `'incremental'`（`detail` 形如 `tomb=1 append=1`）。判"走了哪条路"要看 `kind`，别只看"没报错" |
| 2 | **参数形态跨层不同**：`SparseBM25.score_all(qtoks)` 收 **token 列表**；`Primolix.query(text)` 收**文本并替你分词**。给内核传字符串会被当成**字符集合**，结果**静默全 0 分**（不报错） |
| 3 | **词表封闭**：新词不能增量进已有索引，默认走**结构级重建**（成本 `O(V+nnz)`，本机约 4–6 s），先用 `oov_terms()` 预检。另一条路线已接进内核（最小档，三阶段全 API）：`mount_segment(...)` 把新词放进**只读段**接住（段带原语，打分用**合并 `df`**，`segment_report()` 给**词表代次**），与全库重建逐位一致（实测 `max\|Δ\| = 0`），**不动主表 `T`**、不写盘；`fold_segment(tag)` 再把段并进主表（一次 `O(nnz)`，折叠前后分数不变）；`unmount_segment(tag)` 可把已挂的段**逐项撤回**（分数逐位回到挂载前；要求**后进先出**，且仅对未折叠段）。**并发下的原子换段已在内核默认解决**（见下"当前版本已修"）；**仍未做**的是**跨进程的段共享与发布**（`tc_shared` 目前只共享主表，段的跨进程发布尚未接线）。**修复前实测：不是原子的** —— 无锁时约 **3%**（多次实测 2.3%–3.7%）的并发读数是撕裂的，其中 **96% 是静默错分**（不报错、给一个既不是旧快照也不是新快照的分数），异常只占少数；**外部串行化得到 0 撕裂**，**读者侧的懒建块不是问题**。**当前版本已修**：内核**默认**加可重入读写锁（`SWAP_LOCK=True`：读者并发、写者独占；实测 **0 撕裂**，代价约 **+3.4 µs/查询**，单线程可关）；门面层（第 8 节）再加一层更粗的锁，让跨调用组合也原子。代价是**查询随段数增长**（实测次线性：30 段约 5 倍）。出处：`research/seg_atomic_probe.py`（该台子未随本发布包提供） |
| 4 | **库的输出走 stderr**；`import primolix.core as core; core.VERBOSE = False` 完全静音（CLI 不受影响） |
| 5 | **`FAST_SCORE` 自 2026-10-03 起默认 `True`**；把它关掉会同时失去 `DN_PRECOMP` / `ADD_AT` / `W_CACHE`，查询也更慢 |
| 6 | **`PreallocCSR` 只支持"行增长"**：手工换过 `T`（列宽变了）之后，必须**清掉分块缓存**（`_tc_blocks=[]; _tc_rows=0`）让它重建 |
| 7 | **词表载体是只读 mmap**（`VocabMmap`）：`term2idx[t] = i` 会抛 `TypeError`，要加词得**重发载体**或加**覆盖层** |
| 8 | **别手工删索引里的文件**（见 `index_format.md` 的必需性表）；要清理就整目录重建 |
| 9 | **两套 id 空间**（若你自带语料）：`vocab.json` 之类的"词表"未必等于索引的列空间，只按字符串走 `term2idx`，别拿外部数字当列号 |
| 10 | **物化层、`.mm/` 与共享 CSC 都要看"失效条件"**：`W_CACHE` 的键含参数与块结构（换参、写入、墓碑都会让它失效）；**`.mm/` 有世代键 `mm_key.txt`**（npz 的名字／mtime／size；不符则**整个 `.mm/` 拒用**，回落到 npz 自带的 `T_data`）；`TC_SHARED` 的键是**主表那一份**（`路径\|mtime\|shape\|nnz`），所以只有**重存索引／换版／主表形状或 nnz 变**才会不再命中，**挂段与卸段不影响它**（共享的 CSC 由主表 `T` 建，而挂/卸段**不动 `T`**）。另有一个为"段参与共享"预留的 `_seg_key()`，**目前没有任何调用点**（别假定它生效）。三者都不是"打开就一劳永逸"，用前先想清失效后谁付重建成本。 |
| 11 | **常驻内存有两个口径**：`private`（私有）与 `RSS`（含共享页）。`mmap` postings 与共享 CSC 会让两者给出**相反**结论 —— 单报任一个都会误导；引用本包数字时请注明口径（README 的表标的是**私有**口径，不含共享页）。 |
| 12 | **近似 top-k 是概率保证，不是等式**（`TOPK_APPROX`，0.2.0 默认开）：阈值 = 采样第 k 大 × `TOPK_SHRINK`，理论上可能高于真第 k 大、**漏掉边界文档**。三种形态**自动回精确路径**（非浮点 / 非正分布 / 正分不足 k），返回仍是"k 个下标、分数降序、值等于全量排序前 k 个"这条老契约。已测范围：1M 中文 · 6,980 查询（`R@10` / `MRR@10` 两臂逐位相同）· `k∈{10,100,1000}` · `frac∈{16,32}`；**未测** 8.84M、dense 分支（`k*10`）、短查询与高重复分布。要确定性行为就 `TOPK_APPROX = False`。 |
| 13 | **`WRITE_VEC`（写路径向量化）默认关，且它只影响成本**：输出与 legacy 逐位相同（`T`/`df`/`dl`/分数全等），但"整批只 append 一次"时**更慢**（要先建表 ≈31 ms @V=63,314），所以写单价必须**按形态两报**：重复批 0.34–0.64 µs/posting，一次性 33.2 vs 23.2 ms。别把它当"开了就快"的开关。 |
| 14 | **`explain()` 的 `status` 是唯一判别字段**：`ok` / `oov` / `no_tf` / `unavailable`。**别用 `contribution == 0` 反推** —— OOV 是 0（它确实没贡献），`no_tf`（这篇里没这个词）与 `unavailable`（取不到 tf）语义不同，后者 `contribution` 是 `None`、`tf` 也是 `None`（**不拿 0 冒充**）。`top` 只限制**有贡献**的词条数，其余三类**必出**。 |

## 6. 常见任务（最短路径）

```python
# 建 / 查
z = Primolix.build("docs/", "docs/.idx")
for h in z.query("稀疏 检索", k=5):
    print(h["rank"], round(h["score"], 3), h["file"], h["line"])

# 增量（看 kind 决定是否需要重建）
rep = z.update("docs/")
print(rep)                       # {'kind': 'incremental', 'detail': 'tomb=1 append=1'}

# 换公式（内核层，不重建；参数字段是属性，赋值即生效）
bm = SparseBM25.load("docs/.idx/bm25.npz")
bm.k1, bm.b = 2.0, 0.3

# 归因（分数按词拆开；只用公开属性 + 文档里的 idf 公式）
# 见 examples/run_examples.py 的 explain()

# 视图（多版本 / A-B）
z.attach_view("v2", vectors); print(z.view_report())
```

## 7. 层次血缘索引（`primolix.lineage`）

本发布包**不包含**这一层（`primolix.lineage` 留在源工作区；它尚无回归测试，未进 CI，因此未随包提供）。

## 附：段式发布（staged publish）

把在线新增**攒成一批**、一批只发布**一次**，是"写入即增量"在高频写入场景下的用法：

    st = idx.stage_begin("batch7")     # 起一批
    st.add("doc-001", "稀疏检索的原语层……")
    st.add_many([("doc-002", "……"), ("doc-003", "……")])
    st.stats()                         # {pending, new_terms, base_V, base_T_rows}
    pub = st.publish()                 # 一次性发布；返回 {tag, rows, V0, V1, N, V}
    st.abort()                         # 丢弃本批（不产生任何持久痕迹）

- 累积期**不修改**索引对象：新文档先落在**只读段**里，发布时挂进查询路径（与全库重建逐位一致）。
- **发布 = 一次挂段**（在写锁内），因此对并发查询是原子的。
- 一批写完可用 `idx.compact()` 把段折进主表（查询恢复单块）；`idx.save(fold_first=True)`
  会先折叠再写入磁盘。段与旁车目录 `<索引目录>/.segments/` 可以安全删除，代价是重新发布那些批次。
- 与 `idx.add()` 的分工：`add()` 是**就地**增量（立即改主表）。**0.2.1 起**段在场时 `add()`
  不再抛错：它**先自动折回主表**（`compact()` 那一步，耗时如实报在返回的 `folded_tags` /
  详情里），再走就地增量；返回的 `kind` 仍是 `incremental`。**纯删除同理**（墓碑要改
  `df`/`avgdl`，段在场时写不回盘 ⇒ 不折就会在重载后复活）。
  ⚠️ 想保住"一次挂段、多批追加"的写法，请继续用 `stage_begin()`。
- 段不参与物化层（`W_CACHE`）：批次内查询走现算，折叠后的第一次查询重建一次物化层。
  因此批次越短，现算占比越低。

## 8. 一句话门面（`primolix.api`，**推荐入口**）

> 内核对研究方便、当库用太啰嗦（段要自己写三个 `.npy`、视图要对齐 `meta` 位置、开关是类属性、
> 换公式要摸 `bm.k1`）。这一层**不改内核**，只把常用动作收成对象方法，并补上两个内核没有的默认。

```python
import primolix

idx = primolix.build("docs/", "docs/.idx")   # 建索引（返回门面对象）
idx = primolix.open("docs/.idx")             # 打开已有索引（不存在就报错，不猜）

idx.search("稀疏 检索", k=5)                  # 文本进、结果出（自动分词）
idx.add("docs/")                             # 增量（词表外则如实回退重建，kind='rebuild'）
idx.tune(k1=2.0, b=0.3)                      # 换公式：不重建、不动物料
idx.explain("稀疏 检索")                      # 归因：分数按词拆开
idx.diagnose("稀疏 检索")                     # 诊断：为什么没搜到 / 为什么只有这些（0.2.0）
idx.stats()                                  # 规模 / 开关 / 段数 / 视图数
idx.compact()                                # 压实：把已挂段并进主表，查询恢复单块
idx.save(w_cache=True)                       # 写入磁盘（可选把物化层 w 一起写）
```

| 成员 | 说明 |
|---|---|
| `open(path)` · `open_index(...)` · `build(root, out, ...)` · `build_index(...)` | 入口；`build(..., w_cache=True)` 时建库会把物化层 `w` 一并写入磁盘 |
| **并发安全默认开** | 门面持一把**读写锁**（`RWLock`）：`search`/`stats`/`explain`/`fuse` 走**共享**侧，`add`/`rebuild`/`tune`/段三阶段/`compact`/视图写入/`save` 走**独占**侧，所以读者之间并发、写者与读者互斥，**同一对象被多线程同时用，不会拿到错分数**（依据：无锁时约 **3%**（多次实测 2.3%–3.7%）的并发读数撕裂，其中 **96% 是静默错分**） |
| 归因与诊断 | `explain(q, docid=, k=1, top=6, diagnose=False)`：`status` 四态（见第 5 节 #14）；`diagnose=True` 返回 `dict(contributions=..., diagnosis=...)`，**默认仍返回 `list`**（不破坏既有调用方）· `diagnose(q, k=10)`：`status`（`ok`/`partial_oov`/`all_oov`/`no_positive`/`no_tokens`）＋ `terms`（逐词 `in_vocab`/`df`/`df_ratio`/`idf`）＋ `scores` ＋ `messages` · 位置定位用 `primolix.span`（第 4 节） |
| 段 | `mount(seg_dir, tag, new_terms, dl, rows=None, generation=None)` · `unmount(tag)` · `fold(tag)` · `segments()` · `stage_begin()`（其 `add(docid, text_or_tokens, file=, line=)` 自 0.2.0 起可给段内文档补**真实路径与行号**；缺省用 `docid` 当 `file`、`line=1`，**不编造路径**） |
| 视图 | `views()` · `attach_view(name, vectors, positions=None, ...)` · `detach_view(name)` · `fuse(q, qvec, name, alpha=0.35)` · `save_view` / `load_views` |
| 参数与开关 | `tune(k1=, b=, persist=)` · `stats()`（含各开关当前值）· `set_switches(**kw)`（类属性，进程级） |
| 写入磁盘 | `save(path=None, w_cache=False)`：缺省写回索引目录；给 `path` 则只把**内核索引**另存一份 |

**边界（如实）**：

1. **门面不掩盖内核的边界**：视图族与段各自的"未进 CI / 需自证"照旧适用。内核**自己**已带并发锁
   （`SWAP_LOCK`，见第 4 节），门面这层加的是**跨调用组合**的原子性（`add()`＝墓碑＋追加＋写入磁盘、
   `compact()`＝折段＋清块＋重建），所以直接调内核也不会拿到错分数，但**多步组合**要自己包一层。
2. **开关是类属性，所以是进程级的**：`set_switches(W_CACHE=True)` 影响本进程**所有**索引，且不写入磁盘。
3. 门面的写操作会**等所有读者退出**（读写锁语义），所以单对象高频换段时读写会互相等待；
   这是换正确性的代价。**策略是写者优先**（读者让路：读者可以被推迟、可以滞后，但**不会读到撕裂状态**）；
   要让**写者不再等读者**，需要读侧快照或段清单指针交换（见第 3 节的段说明；
   相关台子 `research/seg_atomic_probe.py` 未随本发布包提供）。
