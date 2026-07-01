# 面向课程资料检索的本地 RAG Agent 系统：阶段计划

## 1. 项目定位

这个项目不是一个单纯的 RAG Demo，而是一个以 Agent 为主体的课程资料问答系统。

RAG 在系统中是 Agent 可以调用的核心工具：它负责从本地课程资料中找证据；Agent 负责判断用户意图、选择工具、组织检索结果、生成回答、检查回答是否有依据。

一句话链路：

用户问题 -> 意图识别 -> 工具路由 -> 文档检索 -> 上下文压缩 -> 生成回答 -> 自检 -> 返回答案和引用

## 2. 为什么这样拆

### Agent 层

- 为什么需要：用户请求不一定都是问答，也可能是导入文档、删除文档、查看统计、重新检索。
- 输入：用户自然语言请求，以及可选参数。
- 输出：工具调用决策、最终回答、引用和执行轨迹。
- 位置：完整链路的调度中心。

### RAG 工具层

- 为什么需要：LLM 本身不知道你的本地课程资料，必须先检索可引用的证据。
- 输入：查询问题、top_k、检索模式。
- 输出：相关 chunk、分数、来源元数据。
- 位置：Agent 调用的核心工具。

### 文档入库层

- 为什么需要：PDF/Markdown/TXT 需要被解析、切块、向量化并写入索引。
- 输入：本地文件或目录。
- 输出：文档记录、chunk 记录、向量索引、BM25 索引。
- 位置：问答前的知识库构建环节。

### API 与 CLI 层

- 为什么需要：CLI 适合开发调试和面试演示；FastAPI 适合展示工程交付能力。
- 输入：命令行参数或 HTTP 请求。
- 输出：导入结果、统计信息、问答结果。
- 位置：系统对外入口。

### Benchmark 层

- 为什么需要：简历里的效果指标必须能本地复现，不能编造。
- 输入：小规模 QA 评测集。
- 输出：Recall@K、Precision@K、拒答率、平均检索耗时。
- 位置：验证检索策略是否真的有效。

## 3. 目标目录结构

```text
myAgent/
  README.md
  pyproject.toml
  .env.example
  docs/
    00_project_plan.md
    architecture.md
    interview_qa.md
    resume_project.md
  data/
    sample_docs/
    eval/
      qa_eval.jsonl
  storage/
    chroma/
    bm25/
  src/
    course_rag_agent/
      __init__.py
      config.py
      schemas.py
      document_loader.py
      text_splitter.py
      embeddings.py
      vector_store.py
      bm25_store.py
      retriever.py
      compressor.py
      tools.py
      agent_graph.py
      api.py
      cli.py
  scripts/
    benchmark_retrieval.py
  tests/
```

说明：当前阶段只创建骨架，后续每个模块会逐步填充代码。

## 4. 分阶段实现计划

### Stage 0：项目结构与总设计

目标：明确系统边界、目录结构、阶段计划。

小实验：查看目录结构，确认项目从空骨架开始。

### Stage 1：基础配置、数据模型与示例资料

目标：建立配置读取、核心数据结构、少量课程示例文档。

会实现：

- `config.py`：读取路径、模型名、检索参数。
- `schemas.py`：定义 Document、Chunk、RetrievalResult 等数据模型。
- `data/sample_docs/`：放入可本地测试的课程资料。

小实验：运行一个 Python 命令，打印配置和示例数据路径。

### Stage 2：文档加载与 chunk 切分

目标：支持 PDF/Markdown/TXT 文档读取，并切成可检索片段。

会实现：

- `document_loader.py`
- `text_splitter.py`

小实验：导入一个目录，输出文档数量、chunk 数量和前几个 chunk。

### Stage 3：Embedding、Chroma 与 BM25 入库

目标：把 chunk 同时写入向量索引和关键词索引。

会实现：

- `embeddings.py`
- `vector_store.py`
- `bm25_store.py`

小实验：对同一个问题分别跑向量检索和 BM25 检索。

### Stage 4：混合检索与低相关度拒答

目标：合并向量分数与 BM25 分数，输出可解释的检索结果。

会实现：

- `retriever.py`
- 低相关度阈值
- 引用元数据

小实验：对相关问题返回引用；对无关问题触发拒答。

### Stage 5：上下文压缩与回答生成

目标：控制传给 LLM 的上下文长度，减少无关信息。

会实现：

- `compressor.py`
- 基于证据的回答 prompt
- 引用格式

小实验：给出一个问题，返回答案、引用和检索分数。

### Stage 6：LangGraph Agent 状态流

目标：让 RAG 成为 Agent 可调用工具，而不是直接写死问答链。

会实现：

- `tools.py`
- `agent_graph.py`
- intent、tool routing、retrieve、compress、generate、self check
- 最大重试次数控制

小实验：打印一次完整 tool trace。

### Stage 7：CLI 与 FastAPI

目标：提供可演示的命令行和 HTTP 接口。

会实现：

- `cli.py`
- `api.py`
- `/documents/import-dir`
- `/documents/upload`
- `/documents/{doc_id}`
- `/stats`
- `/ask`

小实验：用 CLI 导入资料，用 Swagger 调用 `/ask`。

### Stage 8：Benchmark 与面试材料

目标：用小规模人工标注评测集对比纯向量、纯 BM25、混合检索。

会实现：

- `data/eval/qa_eval.jsonl`
- `scripts/benchmark_retrieval.py`
- `docs/interview_qa.md`
- `docs/resume_project.md`

小实验：输出 Recall@K、Precision@K、拒答率、平均检索耗时。

## 5. 当前轻量版本边界

- 先做本地单机版本，不做多用户权限、在线任务队列和分布式索引。
- 先用小规模课程资料验证链路，不声称线上用户量或生产效果。
- 自检先从规则和检索证据一致性开始，后续可升级为 LLM judge。
- 上下文压缩先做排序、去重和长度截断，后续可升级为模型摘要式压缩。
- 混合检索先做分数归一化和加权融合，后续可升级为 reranker。

## 6. 面试主线

可以这样介绍：

我做的是一个面向课程资料的本地 RAG Agent。系统把 RAG 封装成 Agent 的工具，由 LangGraph 控制意图识别、工具路由、检索、上下文压缩、生成和自检。文档侧支持 PDF、Markdown、TXT 入库，检索侧对比了向量检索、BM25 和混合检索，并用小规模人工标注 QA 集合记录 Recall@K、Precision@K、拒答率和检索耗时。项目重点不是堆功能，而是保证每个回答都有本地资料证据和可复现评测。

