# Phase 5 Hybrid Retrieval And Context Compression

本阶段增强检索融合和上下文压缩，保持轻量实现，不引入 reranker 或额外 tokenizer 依赖。

## Fusion Strategies

默认策略仍是：

```text
weighted_score_fusion
```

计算方式：

```text
final_score = vector_weight * vector_score
            + (1 - vector_weight) * bm25_confidence
```

其中 BM25 原始分数会先转成 0-1 附近的 confidence。

新增策略：

```text
rrf_fusion
```

RRF 使用向量检索排名和 BM25 排名做 Reciprocal Rank Fusion：

```text
raw_score = 1 / (rrf_k + vector_rank) + 1 / (rrf_k + bm25_rank)
```

项目中会把 RRF 分数归一化到 0-1 附近，避免破坏现有 `SCORE_THRESHOLD` 的语义。

配置：

```text
FUSION_STRATEGY=weighted_score_fusion
RRF_K=60
VECTOR_WEIGHT=0.65
```

## Standard Retrieval Hit

Agent 中的 `retrieval_hits` 统一为：

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "source": "...",
  "text": "...",
  "vector_score": 0.52,
  "bm25_score": 1.37,
  "final_score": 0.45,
  "score": 0.45,
  "rank": 1,
  "start_pos": 0,
  "end_pos": 100,
  "chunk_index": 0,
  "metadata": {}
}
```

`score` 暂时保留为 `final_score` 的兼容别名。

## Context Compression

`compress_node` 规则：

```text
按 final_score 降序排序
按 chunk_id 去重
使用字符预算，不强行引入 tokenizer
同一 source 的相邻 chunk 合并成一个 context group
每个 chunk 保留 citation_id
输出 compressed_context、citations、citation_map
```

配置：

```text
CONTEXT_CHAR_BUDGET=4500
```

`citation_map` 示例：

```json
{
  "1": {
    "source": "...",
    "chunk_id": "...",
    "final_score": 0.45
  }
}
```

## Trace Fields

`retrieve_node` 新增：

```text
fusion_strategy
vector_hit_count
bm25_hit_count
merged_hit_count
final_hit_count
```

`compress_node` 新增：

```text
compression_ratio
context_char_count
final_hit_count
```

## Tests

```powershell
D:\Anaconda\envs\myAgent\python.exe -m unittest tests.test_retrieval_phase5
```

完整测试：

```powershell
D:\Anaconda\envs\myAgent\python.exe -m unittest tests.test_self_check tests.test_retrieval_phase5
```

真实 smoke test：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "RAG系统为什么需要检索增强生成？"
```

临时切 RRF：

```powershell
$env:FUSION_STRATEGY='rrf_fusion'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "RAG系统为什么需要检索增强生成？"
Remove-Item Env:FUSION_STRATEGY
```
