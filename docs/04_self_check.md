# Phase 4 Self Check

本阶段在 `generate_node` 后加入 `self_check_node`，对答案做证据支持检查。

## Target Flow

```text
compress
-> generate
-> self_check
   ├─ pass -> final
   ├─ fail and generation_retry_count < max_generation_retries -> generate
   └─ fail and generation_retry_count >= max_generation_retries -> final
```

## New State Fields

```text
self_check_passed        自检是否通过
self_check_reason        自检失败或通过原因
unsupported_claims       不被支持的声明或规则失败项
citation_coverage        inline citation 的有效覆盖率
groundedness_score       规则化证据支持分数
generation_retry_count   因 self-check 失败而重新生成的次数
max_generation_retries   最大重新生成次数，默认 1
```

## Rule-Based Self Check

第一版只做规则检查，不引入 LLM Judge，保证稳定和可复现。

规则包括：

```text
answer 不能为空
refused=False 时 citations 不能为空
非拒答答案必须包含 [1] 这类 inline citation
inline citation id 必须在 citations 范围内
citations 的 source/chunk_id 必须来自 retrieval_hits
如果答案说“根据资料/文档显示”等，但 citations 为空，判失败
top_score 低于阈值时，不允许强行通过
```

## LLM Judge

当前未加入 LLM-as-judge。

原因：

```text
Phase 4 先保证自检稳定、可解释、可本地复现
LLM Judge 会增加延迟、成本和 JSON 解析失败风险
后续可作为可选增强加入，并保留规则 fallback
```

## Routing Rules

`self_check_node` 后使用 `add_conditional_edges`：

```text
self_check_passed=True
  -> final

self_check_passed=False and generation_retry_count < max_generation_retries
  -> generate

self_check_passed=False and generation_retry_count >= max_generation_retries
  -> final
```

如果状态已经 `refused=True`，也会直接进入 `final`，避免空上下文等失败状态反复生成。

## Regenerate

如果 self-check 失败且仍可重试：

```text
self_check -> generate
```

`generate_node` 会：

```text
generation_retry_count += 1
把 self_check_reason 写入 prompt
要求模型修复引用缺失或无证据结论
```

## Tests

规则单测：

```powershell
D:\Anaconda\envs\myAgent\python.exe -m unittest tests.test_self_check
```

真实 Agent smoke test：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli agent "RAG系统为什么需要检索增强生成？"
```

预期 trace 包含：

```text
generate_node
self_check_node
final_node
```

self-check trace 示例：

```json
{
  "node": "self_check_node",
  "action": "rule_self_check",
  "decision": "pass",
  "next_node": "final",
  "reason": "ok",
  "self_check_passed": true,
  "citation_coverage": 1.0,
  "groundedness_score": 1.0,
  "unsupported_claims": [],
  "generation_retry_count": 0,
  "max_generation_retries": 1
}
```
