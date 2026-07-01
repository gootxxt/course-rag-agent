from __future__ import annotations

from typing import Any

from .config import Settings, settings
from .knowledge_base import KnowledgeBase
from .qa import QAService


def get_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Search local course documents and return relevant chunks with scores and sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User question or search query."},
                        "top_k": {"type": "integer", "description": "Number of final chunks to return."},
                        "candidate_k": {"type": "integer", "description": "Number of candidates before fusion."},
                        "score_threshold": {"type": "number", "description": "Minimum final relevance score."},
                        "mode": {"type": "string", "enum": ["vector", "bm25", "hybrid"]},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_ask",
                "description": "Answer a question using local retrieved evidence and citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "candidate_k": {"type": "integer"},
                        "score_threshold": {"type": "number"},
                        "mode": {"type": "string", "enum": ["vector", "bm25", "hybrid"]},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_file",
                "description": "Import one txt, md or pdf file into the local knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "strategy": {"type": "string", "enum": ["recursive", "semantic"]},
                        "chunk_size": {"type": "integer"},
                        "overlap": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_dir",
                "description": "Import all supported txt, md and pdf files from a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "strategy": {"type": "string", "enum": ["recursive", "semantic"]},
                        "chunk_size": {"type": "integer"},
                        "overlap": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_file",
                "description": "Delete an existing document by path-derived doc_id and import the file again.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "strategy": {"type": "string", "enum": ["recursive", "semantic"]},
                        "chunk_size": {"type": "integer"},
                        "overlap": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_doc",
                "description": "Delete all chunks of a document by doc_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"doc_id": {"type": "string"}},
                    "required": ["doc_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stats",
                "description": "Return knowledge base document and chunk statistics.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]


def get_tool_names() -> list[str]:
    return [tool["function"]["name"] for tool in get_tool_schemas()]


class ToolExecutor:
    def __init__(self, kb: KnowledgeBase | None = None, app_settings: Settings = settings):
        self.settings = app_settings
        self.kb = kb or KnowledgeBase(app_settings)
        self.qa_service = QAService(self.kb, app_settings)

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        if name == "rag_search":
            hits = self.kb.search(
                query=arguments["query"],
                top_k=arguments.get("top_k"),
                candidate_k=arguments.get("candidate_k"),
                score_threshold=arguments.get("score_threshold"),
                mode=arguments.get("mode", "hybrid"),
            )
            return {"hits": [hit.model_dump() for hit in hits]}

        if name == "rag_ask":
            response = self.qa_service.ask(
                query=arguments["query"],
                top_k=arguments.get("top_k"),
                candidate_k=arguments.get("candidate_k"),
                score_threshold=arguments.get("score_threshold"),
                mode=arguments.get("mode", "hybrid"),
            )
            return response.model_dump()

        if name == "add_file":
            result = self.kb.add_file(
                arguments["path"],
                strategy=arguments.get("strategy"),
                chunk_size=arguments.get("chunk_size"),
                overlap=arguments.get("overlap"),
            )
            return result.model_dump()

        if name == "add_dir":
            result = self.kb.add_dir(
                arguments["path"],
                strategy=arguments.get("strategy"),
                chunk_size=arguments.get("chunk_size"),
                overlap=arguments.get("overlap"),
            )
            return result.model_dump()

        if name == "update_file":
            result = self.kb.update_file(
                arguments["path"],
                strategy=arguments.get("strategy"),
                chunk_size=arguments.get("chunk_size"),
                overlap=arguments.get("overlap"),
            )
            return result.model_dump()

        if name == "delete_doc":
            return self.kb.delete_doc(arguments["doc_id"])

        if name == "stats":
            return self.kb.stats()

        raise ValueError(f"Unknown tool: {name}")

