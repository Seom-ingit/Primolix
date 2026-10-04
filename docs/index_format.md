# 索引格式（`primolix`）

*English: [index_format.en.md](index_format.en.md)*

> 本文的结论来自一个可以直接重跑的实验：建一个小索引，列出全部文件，再逐个删掉，试还能不能查。
> 环境：本机 Python 3.13、4 篇文档的小语料。`fmt` = `primolix-bm25-v3`。

## 1. 目录长什么样

```
<索引目录>/
├─ bm25.npz                       ← 原语本体（唯一的"真数据"）
├─ bm25.npz.mm/                   ← postings 的 mmap 物化（派生）
│   ├─ T_data.npy  T_idx.npy  T_ptr.npy  T_shape.npy
│   ├─ dl.npy  idf.npy  dead.npy
│   ├─ mm_key.txt                 ← **世代键**：npz 的「名字|mtime|size」（**不符则整个 `.mm/` 拒用**）
│   ├─ w_0.npy … w_k.npy          ← 物化层（仅 `--w-cache` / `save(w_cache=True)` 时存在）
│   └─ w_key.txt                  ← 物化层的内容型键（装载时比对，不符**拒用**）
├─ .segments/                      ← 段式发布的派生件（仅用过 `stage_*` 时存在；整个目录可安全删除）
│   ├─ manifest.jsonl              ← 追加式事件日志（`publish` / `folded_all` / `reset`）
│   └─ <tag>/                      ← 段的三件套 ＋ `dl.npy`（详见本文最后一节）
├─ bm25.npz.vocab/                ← 词表载体（mmap 友好，派生）
│   ├─ vocab.blob  vocab_offs.u32  vocab_sorted.u32
│   ├─ vocab_hot.u32              ← 热点词（可选）
│   └─ vocab_key.txt              ← **世代键**：npz 的「名字|mtime|size」（**不符、缺键或长度不符则整个载体拒用**）
├─ meta.jsonl                     ← 每个 chunk 的元数据（file / line / text / row）
└─ hashes.json                    ← 源文件 md5（供 `update` 判断改动）
```

`.mm/` 与 `.vocab/` 都是**派生件**：删掉只会让下次装载重建 / 退化，不影响正确性（`bm25.npz` 才是唯一不可丢的）。物化层默认**不写**；开启了才多出 `w_*.npy` 与 `w_key.txt`，
代价是盘上 `+nnz×4 B`，收益是冷启动免重建。

**世代键的语义**：`mm_key.txt` 把 `.mm/` 钉在某一个 `bm25.npz` 上（名字／mtime／大小）。不符、缺失
或内容被截断就**整个 `.mm/` 拒用**，改从 npz 里的 `T_data` 读（**结果等价**，只是不走 mmap），
于是"写入磁盘写到一半"的任何残留都是**失败安全**的，不会出现"新 npz 配旧 postings"的静默错分数。
升级提示：**加键之前**写入磁盘的索引其 `.mm/` 没有键，**首次加载会拒用**（正确、稍慢、打一行 WARN）；
**重存一次**即恢复 mmap 路径。

**词表载体的世代键**：`vocab_key.txt` 把 `.vocab/` 钉在同一个 `bm25.npz` 上（名字／mtime／大小，尾部另带
词表长度便于人读）。载体与 `bm25.npz` **不同代**、**缺键**，或**长度与 npz 自带的词表不一致**时，一律
**拒用载体**，回落到 **npz 自带的词表**（`vocab` 键永远在场，结果等价，只是不走载体的 mmap）——
于是"就地重存时侧车写入失败"这类残留不会让新词静默消失。

**多进程部署的实操结论（实测）**：读者进程一旦 mmap 了 `.mm/`，写者**就地重存**只能换掉 `bm25.npz`，
`.mm/` 与 `.vocab/` 会**写不进去**（Windows 报 `OSError(22)`，库会打 `[WARN]` 留痕），所以要发布新世代，
请**写到新目录**（或在没有读者的时间窗内重存）；而"半发布"的历史目录是**安全**的 ——
新读者会因世代键不符**拒用旧 `.mm/`**，直接从 npz 读到一个**完整**的世代（只是不走 mmap）。

`bm25.npz` 里装的键（实测）：

| 键 | dtype / 形状 | 含义 |
|---|---|---|
| `fmt` | 字符串 | `primolix-bm25-v3`（格式版本） |
| `T_data` / `T_idx` / `T_ptr` | float32 / int32 / int32 | **原语 `tf`** 的 CSR 三件套 |
| `T_shape` | int64 ×2 | `(行=文档数, 列=词表大小)` |
| `df` | int32 × V | 文档频 |
| `dl` | float32 × N | 文档长度 |
| `_idf` | float32 × V | 保存时的 idf（查询期可重算） |
| `dead` | bool × N | **墓碑**位图（增量写入后旧行为 True） |
| `N_live` | 标量 | 活跃文档数 |
| `params` | float32 ×3 | `k1` / `b` / `avgdl` |
| `vocab` | object × V | 词表（词 → 列） |
| `W_shape` | int64 ×2 | 兼容用（旧版 v1/v2 才落 `W` 分数矩阵） |

一句话：**`bm25.npz` 是可逆原语，其余都是可以从它重建的物化或清单。**

## 2. 哪些文件是必需的（实测：逐个删掉再查）

| 删掉的部分 | 结果（本次 1 条查询） |
|---|---|
| `bm25.npz.mm/T_idx.npy` · `T_ptr.npy` | 失败：`FileNotFoundError` |
| `bm25.npz.mm/idf.npy` | 失败：`TypeError: 'NoneType' object is not subscriptable`（报错不可读） |
| `bm25.npz.vocab/`（三个文件） | 失败：`FileNotFoundError` |
| `bm25.npz.vocab/vocab_key.txt` | 未单独删测：按规则**拒用载体**，回落 npz 自带的词表（结果等价，只是不走载体的 mmap） |
| `meta.jsonl` | 失败：`FileNotFoundError` |
| `bm25.npz.mm/dl.npy` | 本次通过 |
| `bm25.npz.mm/T_data.npy` | 本次通过 |
| `bm25.npz.mm/T_shape.npy` | 本次通过 |
| `hashes.json` | 本次通过（`update` 会失去"哪些文件变了"的依据） |

> **"本次通过"是弱证据**：只跑了一条查询、一个 4 篇文档的小索引。它说明"这条路径不读它"，
> 不等于"可以删"。稳妥做法：**不要手动删任何文件**；要清理就用重建。

## 3. 可移植性

- **把整个目录拷到别处仍可查询**（目录内是相对引用，没有硬编码绝对路径）。
- 只读挂载、共享只读也是安全的：查询只读，`update` 才会写。
- 复制时**整目录一起拷**（缺件会按第 2 节的报错失败）。

## 4. 版本与兼容

- 版本写在 `bm25.npz` 的 `fmt` 键：`primolix-bm25-v3`（当前）。
- **改名前的索引（`fmt = zelix-bm25-v3`）仍可加载**：读取时会把旧格式串归一化到新名，
  盘上格式一字未变（改的只是包名与类名），因此升级不需要重建索引。
- 旧格式（`v1` / `v2`，含算好的 `W_data` 分数矩阵）**仍可加载**：加载后恢复旧的快路径；
  新索引不再落 `W`（分数是导出物）。
- **没有**跨大版本的迁移工具：换大版本请重建索引（重建结果是确定的，可以逐位对账）。

## 5. 重建与清理

- **重建**：`python -m primolix index <目录> --out <索引>`（旧目录会被覆盖）。
- **只想省盘**：删掉整个索引目录再重建，比"删其中几个文件"安全（见第 2 节的说明）。
- **增量**：`python -m primolix update <目录> --out <索引>`（按 `hashes.json` 判断改动；改一篇只做那一篇的工作）。

## 6. 已知的报错可读性问题（待改）

- 缺 `idf.npy` 报 `TypeError: 'NoneType' object is not subscriptable`；
- 缺其它件报裸 `FileNotFoundError`（不带上下文）。

期望行为：加载时先做一次完整性预检，缺件就报一句
"索引不完整：缺 `X`；可用 `index` 重建或补回该文件"。

## 附：段式发布的派生件 `<索引目录>/.segments/`（可删可重建）

    <idx>/.segments/manifest.jsonl          追加式事件日志
    <idx>/.segments/<tag>/<tag>.data.npy    段的三件套（CSC）
    <idx>/.segments/<tag>/<tag>.idx.npy
    <idx>/.segments/<tag>/<tag>.ptr.npy
    <idx>/.segments/<tag>/dl.npy            段内每行的文档长度

`manifest.jsonl` **只追加、从不改写**，三种事件：`publish`（**提交点**）、`folded_all`（段已折进主表）、
`reset`（整库重建，之前的事件作废）。载入时按事件顺序回放：把仍在挂的段重挂回去；已折叠、或与当前词表
**不同代**的段会被**跳过并如实报告**，不会静默混用。段在场时不能写入磁盘（主表装不下它们的行）：
`save(fold_first=True)` 会先折叠再写盘。整个 `.segments/` 可以安全删除，代价只是重新发布那些批次。

## 7. 复现本文

1. 用几篇文档建一个小索引：`python -m primolix index <小语料目录> --out <索引目录>`。
2. 列出索引目录下的全部文件。
3. 先备份目录，再每次删掉其中一个文件，然后跑一条
   `python -m primolix query "..." --out <索引目录>`，记下这次是通过还是失败。
4. 把"删掉哪个文件、结果是什么"整理成第 2 节的表。
