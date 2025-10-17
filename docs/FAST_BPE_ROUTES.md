# Fast BPE 两条路线设计说明

目的：在保持可读性与与现有训练行为一致（含 tie-break 规则）的前提下，显著降低 BPE 训练的时间复杂度与常数开销。

现状（朴素版，参考 cs336_basics/basic_bpe.py）
- 每轮：统计所有相邻对频次 O(L)，选最大对，再对所有词形重写一次序列 O(L)。总体近 O(NL)。
- 选对 tie-break：当频次相同，按 `(vocab[a], vocab[b])` 的字节序优先（见 cs336_basics/basic_bpe.py:257–263）。

目标
- 降低每轮工作量：只处理“包含被合并 pair 的位置”，避免全表扫描与重写。
- 保持可重复的合并序：复用/模拟现有 tie-break 语义，保证与朴素实现产出一致。
- 控制内存：避免指数增长的数据结构；必要时采用惰性（lazy）维护策略。

---

路线 A：倒排索引 + 最大堆（保留 tuple 表示）

思路概述
- 仍用 `byte_word_counter: Dict[Tuple[int,...], int]` 存“词形 -> 频数”。
- 建立 pair 的倒排索引：`pair_index[(a,b)] -> {word_tuple: occ_in_word}`，表示每个词形中该 pair 出现次数。
- 用“可增减”的最大堆维护 `pair_count[(a,b)] = Σ counter[word] * occ_in_word`。
- 每轮合并只遍历 `pair_index[(a,b)]` 的相关词形并重写它们，按影响的邻域 pair 增量更新堆与索引。

数据结构
- byte_word_counter: Dict[Tuple[int,...], int]
- pair_index: Dict[Tuple[int,int], Dict[Tuple[int,...], int]]
- pair_count_heap: 支持 increase/decrease/pop_max 的最大堆（可先用 `heapq` + 延迟删除，再迭代优化）
- vocab: Dict[int, bytes]（解码使用与 tie-break 用到 bytes 顺序）

初始化
1) 从语料构建 `byte_word_counter`（已有）。
2) 扫所有词形，统计相邻 pair 频次：
   - 对每个词形 `w`，对每个相邻对 `p`：`pair_index[p][w] += 1`；
   - 同步维护 `pair_count[p] += counter[w]`。
3) 将 `pair_count` 放入最大堆。

训练主循环（直到词表达标）
1) 从堆取当前最高频 pair = (a,b)。若有并列，按 `(vocab[a], vocab[b])` 进行 tie-break（只在必要时比较）。
2) 遍历 `pair_index[(a,b)]` 的所有 `w`：
   - 计算 `new_w = merge_pair(w, (a,b), new_id)`；
   - 用 `counter[w]` 作为权重，对受影响的邻域 pair 做“增量更新”：
     - 移除旧对：`(x,a)`、`(a,b)`、`(b,y)`；加入新对：`(x,new_id)`、`(new_id,y)`；
     - 对 `pair_index` 采用惰性维护：旧映射可保留，使用时校验“该词形是否仍包含该对”，并在遍历后将 `pair_index[(a,b)]` 置空以回收；
     - 对堆采用惰性维护：把增量累积到 `pending_delta[(pair)]`，在下次查询/取堆顶时统一下放。
   - 更新 `byte_word_counter`：`counter[w] -= freq ; counter[new_w] += freq`（若 freq 为 `pair_index[(a,b)][w]`，则可一次性替换）。
3) 记录 merge，写入 `vocab[new_id] = vocab[a] + vocab[b]`。

复杂度与内存
- 初始化 O(L)；每轮只遍历包含目标 pair 的词形集合，通常远小于全量。
- 堆操作近 O(log U)，U 为活跃 pair 数；增量更新 amortized O(1) 邻域数量。
- 额外内存：`pair_index` 规模 ~ O(L)。可在某 pair 完成后清空对应 entry 以回收内存。

tie-break 兼容性
- 维持“先看 count，再看 `(vocab[a], vocab[b])`”的一致性：
  - 堆按 count 排序；发生并列时，对候选堆顶做一次 bytes 比较决定胜者；
  - 避免将 `(vocab[a],vocab[b])` 放入堆键，减少堆调整成本。

风险与边界
- `pair_index` 可能较大，需要控制内存与适时清理；
- 惰性策略需要在访问处做校验，确保不出现“幽灵 pair”；
- Python 层 dict-of-dict 访问开销不小，需注意热路径。

---

路线 B：IndexedList + Multiset（fast_minbpe 风格）

思路概述
- 用双向链表表示每个词形的 token 序列；
- 用懒索引 `stale_index[(a,b)] -> [Occurrence]` 指向“可能是 (a,b) 起点”的节点；
- 用可增减的最大堆 Multiset 维护加权的 pair 频次；
- 合并在链表原地进行，且仅对合并邻域的 pair 计数与索引做增量更新；
- 总体复杂度接近 O(L + N log U)。

数据结构
- Node: `val, prev, next`
- Word: `{head: Node, count: int}`（count 为该词形频次，作为权重）
- Occurrence: `{word: Word, node: Node}` 表示一个 pair 的起始位置
- stale_index: Dict[(a,b), List[Occurrence]]（懒：只追加，不强制清理）
- Multiset: 支持 `increase/decrease/pop_max`，内部记录 `count[(a,b)]`，并允许“懒下放” pending 变更
- vocab: Dict[int, bytes]

初始化
1) 将每个词形构造成链表（按词频构建 Word 对象，但不复制多份）
2) 扫每个词形的相邻 pair：
   - `stale_index[(t[i],t[i+1])].append(Occurrence(word,node_i))`
   - `multiset.increase((t[i],t[i+1]), by=word.count)`

训练主循环（合并 (a,b) → new_id）
1) `pair = multiset.pop_max()`（按需要先应用 pending 增量）；tie-break：如 count 并列，再按 `(vocab[a],vocab[b])` 比较。
2) 遍历 `stale_index[pair]`：
   - 懒校验：若 `node.val!=a` 或 `node.next is None` 或 `node.next.val!=b`，跳过；
   - 原地合并（对该词形所有出现位置都会访问到）：
     - 设局部为 `... x a b y ...`：删除 `node.next`（b），将 `node.val = new_id`；
     - 统计增量：`decrease((a,b), count=word.count)`；对 `(x,a)`、`(b,y)` 做 decrease；对 `(x,new_id)`、`(new_id,y)` 做 increase；
     - 索引补充：将 `Occurrence(word,node.prev)` 追加到 `(x,new_id)`，将 `Occurrence(word,node)` 追加到 `(new_id,y)`；
3) `vocab[new_id] = vocab[a] + vocab[b]`。

复杂度
- 每个真实发生的合并触发 O(1) 次邻域更新；所有合并总次数 ≤ L-1；
- 堆操作 O(log U)；总体逼近 O(L + N log U)。

tie-break 兼容性
- 同路线 A：仅在并列时按 `(vocab[a],vocab[b])` 比较，避免将 bytes 关键信息放入堆键。

风险与边界
- 实现复杂度较高：需要严谨处理节点删除后的迭代安全，避免重复/漏合并；
- 懒索引可能累积过期条目：需要在访问时校验；可在 pair 完成后将其列表丢弃以回收；
- 注意权重（word.count）对统计的影响，确保每次增减按权重应用。

---

两路线对比与选择
- 开发复杂度：A < B；
- 加速潜力：A 显著减少每轮处理的词形数量；B 更进一步，避免重建 tuple、原地合并，理论上更快；
- 兼容性：两者都可复用现有正则切词与 tie-break 逻辑，便于对齐朴素实现；
- 内存：A 的 `pair_index` 更占内存；B 的 `stale_index` 也在 O(L) 级别，但 Occurrence 对象更轻量，且可丢弃完成的 pair 列表。

实施计划与交付
- 路线 A（文件建议）
  - `cs336_basics/fast_bpe_index.py`：实现倒排索引 + 堆的训练器；API 与 decode/encode 接口对齐；
  - `tests/`：小语料一致性测试（与 basic_bpe 输出合并序列一致）、基准测试脚本；
- 路线 B（文件建议）
  - `cs336_basics/fast_bpe_indexedlist.py`：实现 IndexedList + Multiset；
  - `tests/`：与路线 A 相同的等价性与性能回归；
- 验证与基准
  - correctness：对 `tests/fixtures/tinystories_sample.txt`、`tinystories_sample_5M.txt` 校验 merges 与最终 vocab 一致；
  - performance：记录构建耗时、每轮耗时、堆操作次数、内存峰值（可选）。

附：伪代码

路线 A（核心循环）
```
init(byte_word_counter)
build pair_index, pair_count, heap
for step in range(target_vocab - 256):
  pair = heap.pop_max() # tie-break with (vocab[a],vocab[b]) if needed
  new_id = 256 + step; vocab[new_id] = vocab[a] + vocab[b]
  for (word, occ) in pair_index[pair].items():
    freq = counter[word]
    new_word = merge_pair(word, pair, new_id)
    # counter move
    counter[word] -= freq; counter[new_word] += freq
    # update neighbors around occurrences (lazy: accumulate deltas)
    dec((x,a), by=freq), dec((a,b), by=freq*occ), dec((b,y), by=freq)
    inc((x,new), by=freq), inc((new,y), by=freq)
    # pair_index lazy update: add new neighbors; old entries tolerated
  pair_index[pair].clear()
```

路线 B（合并）
```
init words as linked lists; build stale_index and multiset
for step in range(target_vocab - 256):
  pair = multiset.pop_max() # tie-break with (vocab[a],vocab[b])
  new_id = 256 + step; vocab[new_id] = vocab[a] + vocab[b]
  for occ in stale_index[pair]:
    node = occ.node; word = occ.word
    if node.val != a or not node.next or node.next.val != b: continue
    # neighbors: x=node.prev?.val, y=node.next.next?.val
    multiset.decrease((a,b), by=word.count)
    if node.prev: multiset.decrease((node.prev.val,a), by=word.count)
    if node.next.next: multiset.decrease((b,node.next.next.val), by=word.count)
    # in-place merge
    node.next.delete(); node.val = new_id
    # add new neighbors and index
    if node.prev:
      multiset.increase((node.prev.val,new_id), by=word.count)
      stale_index[(node.prev.val,new_id)].append(Occurrence(word,node.prev))
    if node.next:
      multiset.increase((new_id,node.next.val), by=word.count)
      stale_index[(new_id,node.next.val)].append(Occurrence(word,node))
  stale_index[pair].clear()
```

备注
- 两方案都只在并列时做 bytes tie-break，可保持堆/堆节点较轻。
- 完整实现需要处理一些工程细节（如 pending deltas 的应用时机、过期索引的校验与清理）。
