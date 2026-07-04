# Phase 3 Query Rewrite And Retry Loop

本阶段在 Phase 2 条件路由基础上加入 query rewrite 和 retry loop。

## Target Flow

```text
retrieve
-> relevance_check
   ├─ enough evidence -> compress -> generate -> final
   ├─ low relevance and retry_count < max_retries -> rewrite -> retrieve
   └─ low relevance and retry_count >= max_retries -> final
```

## New State Fields

```text
original_query      用户最初的问题，生成答案时仍以它为准
query               当前用于检索的问题，rewrite 后会更新
rewritten_query     最近一次改写后的检索问题
retry_count         已经执行过的 rewrite 次数
max_retries         最大 rewrite 次数，默认 2
rewrite_history     每次改写的 old_query/new_query/reason/top_score
reason              当前失败或拒答原因
```

## Rewrite Prompt

`rewrite_node` 会要求模型：

```text
Rewrite user questions into better search queries for a local course-material RAG system.
Preserve the user's original intent.
Do not broaden the question.
Prefer concrete technical terms, aliases, and likely document keywords.
Return only one rewritten query, without explanation.
```

输入包括：

```text
original_query
current search query
failure reason
retry attempt
low-relevance retrieval summary
```

## Routing Rules

`relevance_check_node` 之后使用 `add_conditional_edges`：

```text
relevance_passed == true
  -> compress

relevance_passed == false and retry_count < max_retries
  -> rewrite

relevance_passed == false and retry_count >= max_retries
  -> final
```

`rewrite_node` 使用普通边回到 `retrieve_node`：

```text
rewrite -> retrieve
```

## Infinite Loop Guard

所有 retry 都依赖 `retry_count`：

```text
initial retry_count = 0
rewrite_node 每执行一次 retry_count += 1
retry_count >= max_retries 后必须进入 final
```

默认：

```text
MAX_RETRIES=2
```

## Demo

正常问题，不触发 rewrite：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "RAG系统为什么需要检索增强生成？"
```

模糊问题，触发 rewrite 后成功：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:SCORE_THRESHOLD='0.55'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "context是什么"
Remove-Item Env:SCORE_THRESHOLD
```

无关问题，达到 retry 上限后拒答：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:SCORE_THRESHOLD='0.55'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "火星基地预算是多少"
Remove-Item Env:SCORE_THRESHOLD
```

预期 trace 能看到：

```text
retrieve_node
relevance_check_node
rewrite_node
retrieve_node
relevance_check_node
```

如果最终仍低相关，会看到：

```text
reason = low_relevance_after_retry
refused = true
```
