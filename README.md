# Local Course RAG Baseline

这是一个面向课程资料检索问答的本地 RAG 基线版本。

当前版本先覆盖基础能力：文档加载、chunk 切分、本地 embedding、Chroma 向量库、BM25、混合检索、低相关度拒答、引用溯源、CLI 和 FastAPI。复杂 LangGraph Agent、Function Calling、自检和 benchmark 会在下一阶段实现。

## 1. Install

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

默认 embedding 模型是 `BAAI/bge-m3`。如果本机暂时没有模型文件或下载较慢，代码会启用 hash embedding fallback，方便先跑通 smoke test。正式演示时建议提前下载好 `BAAI/bge-m3` 或改成其他 `sentence-transformers` 模型。

## 2. Configure LLM

编辑 `.env`：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

如果不配置 `OPENAI_API_KEY`，`ask` 仍会返回检索证据摘要，但不会调用真实大模型生成。

## 3. CLI Usage

```powershell
rag-agent reset
rag-agent add-dir data/sample_docs --strategy recursive --chunk-size 500 --overlap 80
rag-agent stats
rag-agent ask "RAG 系统为什么需要 BM25？" --top-k 3
rag-agent ask "这份资料里有没有讲量子通信实验？" --top-k 3
```

也可以不用安装脚本入口，直接运行：

```powershell
python -m course_rag_agent.cli stats
```

## 4. FastAPI

```powershell
uvicorn course_rag_agent.api:app --reload --port 8000
```

打开：

```text
http://127.0.0.1:8000/docs
```

主要接口：

- `POST /documents/upload`
- `POST /documents/import-dir`
- `DELETE /documents/{doc_id}`
- `GET /stats`
- `POST /ask`

## 5. Smoke Test

```powershell
rag-agent reset
rag-agent add-dir data/sample_docs --strategy semantic --chunk-size 500 --overlap 50
rag-agent stats
rag-agent ask "混合检索如何结合向量检索和 BM25？" --mode hybrid --top-k 3
rag-agent ask "课程资料有没有说明火星基地建设预算？" --top-k 3 --threshold 0.4
```

预期结果：

- `add-dir` 显示加载了示例文档并写入 chunk。
- `stats` 显示 documents 和 chunks 大于 0。
- 相关问题会返回答案、citations 和 retrieval_hits。
- 无关问题在阈值较高时会拒答。

## 6. Directory

```text
myAgent/
  requirements.txt
  pyproject.toml
  .env.example
  README.md
  data/
    sample_docs/
      rag_basics.md
      agent_notes.txt
  storage/
    chroma/
    bm25/
  src/
    course_rag_agent/
      config.py
      schemas.py
      utils.py
      document_loader.py
      text_splitter.py
      embeddings.py
      vector_store.py
      bm25_store.py
      retriever.py
      knowledge_base.py
      llm.py
      qa.py
      cli.py
      api.py
```

## 7. Module Responsibilities

- `config.py`：集中读取 `.env` 配置。
- `schemas.py`：定义文档、chunk、检索结果、问答响应。
- `document_loader.py`：读取 txt、md、pdf。
- `text_splitter.py`：实现 recursive 和 semantic 两种切分策略。
- `embeddings.py`：调用本地 `sentence-transformers`，必要时 fallback。
- `vector_store.py`：管理 Chroma collection、metadata、去重、删除和统计。
- `bm25_store.py`：维护本地 BM25 索引。
- `retriever.py`：融合向量检索和 BM25，支持阈值过滤。
- `knowledge_base.py`：对外提供 add/update/delete/reset/stats/search。
- `llm.py`：调用 OpenAI-compatible chat completion。
- `qa.py`：执行检索、拒答、生成、引用返回。
- `cli.py`：命令行入口。
- `api.py`：FastAPI 接口。

## 8. Next Stage

基线跑通后，下一阶段补 Agent 能力：

- LangGraph 状态流。
- Function Calling 工具调度。
- 意图识别。
- 低相关度查询改写和再检索。
- 上下文压缩。
- self_check 自检。
- Recall@K / Precision@K / 拒答率评测。
- 纯向量、纯 BM25、混合检索消融实验。
## Local Evaluation

The project includes a lightweight local eval script. It does not require LangSmith or RAGAS.

Prepare sample documents:

```powershell
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli add-dir data/sample_docs
```

Run eval:

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\Anaconda\envs\myAgent\python.exe scripts/eval_rag_agent.py
```

Default eval data:

```text
data/eval/qa_eval.jsonl
```

Each JSONL sample has:

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

Generated reports:

```text
data/eval/reports/eval_report.json
data/eval/reports/failed_cases.jsonl
```

Metrics:

```text
refusal_accuracy      Whether refuse / answer decisions match should_refuse.
recall_at_k           Whether expected_sources appear in top-k retrieval hits.
precision_at_k        Fraction of top-k hits matching expected_sources.
citation_coverage     Whether final citations can be traced to retrieval_hits.
self_check_pass_rate  Ratio of answers passing rule-based self-check.
rewrite_trigger_rate  Ratio of samples that entered rewrite_node.
rewrite_success_rate  Ratio of rewritten samples that eventually answered or passed relevance check.
avg_latency_ms        Average end-to-end Agent latency.
p95_latency_ms        95th percentile latency.
```

See `docs/06_local_eval.md` for details.
