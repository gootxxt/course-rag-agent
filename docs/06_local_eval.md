# Phase 6 Local Evaluation

本阶段新增本地评测脚本，用小规模 QA 集评估检索、拒答、引用、自检和重写质量。

## Eval Data Format

默认数据文件：

```text
data/eval/qa_eval.jsonl
```

每行一个 JSON：

```json
{
  "id": "rag_basic_001",
  "query": "RAG系统为什么需要检索增强生成？",
  "expected_answer_keywords": ["RAG", "检索", "生成"],
  "expected_sources": ["rag_basics.md"],
  "should_refuse": false,
  "category": "normal"
}
```

字段说明：

```text
id                         样本唯一标识
query                      用户问题
expected_answer_keywords   期望答案关键词，用于粗略检查答案是否覆盖要点
expected_sources           期望召回的资料来源，支持 source 子串匹配
should_refuse              这题是否应该拒答
category                   normal / low_relevance / no_answer / stats / ambiguous
```

## Metrics

```text
total
  样本总数

answer_count / refusal_count
  最终回答和拒答数量

refusal_accuracy
  should_refuse 样本是否拒答，非拒答样本是否回答

recall_at_k
  expected_sources 是否出现在 top-k retrieval_hits

precision_at_k
  top-k retrieval_hits 中有多少来源匹配 expected_sources

citation_coverage
  最终 citations 是否都能追溯到 retrieval_hits

self_check_pass_rate
  self_check_passed=True 的比例

rewrite_trigger_rate
  触发 rewrite_node 的样本比例

rewrite_success_rate
  触发 rewrite 后最终通过相关性检查或成功回答的比例

avg_latency_ms / p95_latency_ms
  Agent 端到端平均耗时和 p95 耗时
```

## Run

先确保知识库里有评测用资料：

```powershell
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli add-dir data/sample_docs
```

运行评测：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe scripts/eval_rag_agent.py
```

输出文件：

```text
data/eval/reports/eval_report.json
data/eval/reports/failed_cases.jsonl
```

切换 RRF 后再评测：

```powershell
$env:FUSION_STRATEGY='rrf_fusion'
D:\Anaconda\envs\myAgent\python.exe scripts/eval_rag_agent.py
Remove-Item Env:FUSION_STRATEGY
```

## Notes

当前评测是轻量本地版本，不依赖 LangSmith 或 RAGAS。它适合用于小规模可复现实验，后续可以扩展为：

```text
vector / bm25 / weighted / rrf 消融实验
query rewrite 前后对比
self-check 失败案例分析
LangSmith trace 或 RAGAS 指标接入
```
