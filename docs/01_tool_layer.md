# Function Calling 工具层说明

当前阶段不是完整 LangGraph Agent，而是先把已有能力包装成 Agent 可调用的工具。

## 为什么需要工具层

基础 RAG 版本里，CLI 和 FastAPI 都是直接调用 `KnowledgeBase` 或 `QAService`。

下一阶段要做 Agent 时，不能把所有逻辑写死在图节点里。更清晰的做法是：

```text
Agent 判断意图 -> 选择工具 -> 调用工具 -> 把工具结果写回状态
```

所以本阶段新增 `tools.py`，它提供两类内容：

- 工具 schema：告诉 LLM/Agent 有哪些工具、每个工具需要哪些参数。
- ToolExecutor：根据工具名和参数执行真实代码。

## 当前工具

```text
rag_search   检索本地资料，返回 chunks 和分数
rag_ask      检索并生成带引用回答
add_file     单文件入库
add_dir      目录批量入库
update_file  删除旧文档后重新入库
delete_doc   按 doc_id 删除文档
stats        查看知识库统计
```

## 数据流

以 `rag_search` 为例：

```text
ToolExecutor.execute("rag_search", args)
-> KnowledgeBase.search()
-> refresh_indexes()
-> HybridRetriever.search()
-> ChromaVectorStore.search() + BM25Store.search()
-> 返回 hits
```

以 `rag_ask` 为例：

```text
ToolExecutor.execute("rag_ask", args)
-> QAService.ask()
-> KnowledgeBase.search()
-> LLMClient.answer()
-> 返回 answer / citations / retrieval_hits
```

## 调试命令

查看工具 schema：

```powershell
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli tools
```

直接调用 stats 工具：

```powershell
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli call-tool stats
```

直接调用 rag_search 工具：

```powershell
D:\Anaconda\envs\myAgent\python.exe -m course_rag_agent.cli call-tool rag_search --args-file data/eval/rag_search_args.json
```

说明：Windows PowerShell 容易吃掉 JSON 字符串里的双引号，所以复杂参数建议用 `--args-file`。

## 面试讲法

可以这样说：

我把当前 RAG、入库、删除、统计等能力封装成 Function Calling 风格的工具。工具层包含 schema 和 executor 两部分：schema 约束工具名与参数，executor 负责把工具调用转成真实的 `KnowledgeBase` 或 `QAService` 方法。这样下一阶段接 LangGraph 时，图节点只需要做意图识别和工具路由，RAG 就能作为 Agent 的一个工具被调用，而不是写死成固定问答链路。
