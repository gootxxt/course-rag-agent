# LangGraph Agent Phase 2

本阶段把 Phase 1 的固定 RAG 链路升级为基于 `add_conditional_edges()` 的条件路由。

## 当前定位

当前系统更准确地说是一个轻量级 Agentic RAG 原型：

```text
intent
  ├─ stats  -> stats -> final
  ├─ ask    -> retrieve -> relevance_check
  │                         ├─ pass -> compress -> generate -> final
  │                         └─ fail -> final
  └─ unknown/irrelevant -> refuse -> final
```

和 Phase 1 相比，图不再只是固定顺序执行。现在节点执行完后，会根据 `AgentState` 中的字段动态选择下一步。

## 条件路由

`intent` 后的路由：

```text
intent == "stats"   -> stats
intent == "ask"     -> retrieve
其他/空问题          -> refuse
```

`relevance_check` 后的路由：

```text
relevance_passed == true and refused == false -> compress
否则                                           -> final
```

`generate` 后目前仍然直接进入 `final`，为后续 `self_check` 节点预留位置。

## 关键节点

```text
intent_node
  判断用户意图，并在 trace 中记录 decision 和 next_node

stats_node
  查询知识库统计信息

refuse_node
  处理当前 Agent 不支持的请求

retrieve_node
  调用 rag_search 工具，写入 retrieval_hits / hit_count / top_score

relevance_check_node
  按 top_score 和 hit_count 做规则判断，决定去 compress 还是 final

compress_node
  对 hits 去重、排序、截断，生成 compressed_context 和 citations

generate_node
  只基于 compressed_context 调用 LLM 生成答案

final_node
  统一输出 answer / citations / refused / reason / tool_trace
```

## Trace 价值

每个关键节点都会写入 `tool_trace`。Phase 2 重点新增：

```text
decision   当前节点做出的判断
next_node  图接下来要进入的节点
reason     判断依据
```

例如低相关问题会出现：

```json
{
  "node": "relevance_check_node",
  "action": "check_relevance",
  "decision": "refuse",
  "next_node": "final",
  "reason": "top_score=None, threshold=0.95, hit_count=0"
}
```

## Demo

统计问题：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "现在知识库有多少文档和chunk？"
```

预期路径：

```text
intent_node -> stats_node -> final_node
```

正常资料问答：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "RAG系统为什么需要检索增强生成？"
```

预期路径：

```text
intent_node -> retrieve_node -> relevance_check_node -> compress_node -> generate_node -> final_node
```

低相关拒答：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:SCORE_THRESHOLD='0.95'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "火星基地预算是多少？"
Remove-Item Env:SCORE_THRESHOLD
```

预期路径：

```text
intent_node -> retrieve_node -> relevance_check_node -> final_node
```

## 仍未实现

```text
query rewrite
retry loop
LLM self_check
reranker
benchmark
```

这些属于后续阶段。
