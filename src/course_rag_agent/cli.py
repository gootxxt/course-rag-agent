from __future__ import annotations

import argparse
import json

from .agent_graph import run_agent
from .config import settings
from .knowledge_base import KnowledgeBase
from .qa import QAService
from .tools import ToolExecutor, get_tool_schemas


def _print_json(payload) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_tool_args(raw_args: str, args_file: str | None) -> dict:
    if args_file:
        with open(args_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(raw_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-agent", description="Local course RAG baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_chunk_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--strategy", choices=["recursive", "semantic"], default=None)
        p.add_argument("--chunk-size", type=int, default=None)
        p.add_argument("--overlap", type=int, default=None)

    p = sub.add_parser("add-file")
    p.add_argument("path")
    add_chunk_args(p)

    p = sub.add_parser("add-dir")
    p.add_argument("path")
    add_chunk_args(p)

    p = sub.add_parser("update-file")
    p.add_argument("path")
    add_chunk_args(p)

    p = sub.add_parser("delete-doc")
    p.add_argument("doc_id")

    p = sub.add_parser("ask")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--candidate-k", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")

    p = sub.add_parser("agent")
    p.add_argument("query")

    p = sub.add_parser("call-tool")
    p.add_argument("name")
    p.add_argument("--args", default="{}", help="JSON object string for tool arguments")
    p.add_argument("--args-file", default=None, help="Path to a JSON file with tool arguments")

    sub.add_parser("tools")
    sub.add_parser("stats")
    sub.add_parser("reset")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    kb = KnowledgeBase(settings)

    if args.command == "add-file":
        _print_json(kb.add_file(args.path, strategy=args.strategy, chunk_size=args.chunk_size, overlap=args.overlap))
    elif args.command == "add-dir":
        _print_json(kb.add_dir(args.path, strategy=args.strategy, chunk_size=args.chunk_size, overlap=args.overlap))
    elif args.command == "update-file":
        _print_json(kb.update_file(args.path, strategy=args.strategy, chunk_size=args.chunk_size, overlap=args.overlap))
    elif args.command == "delete-doc":
        _print_json(kb.delete_doc(args.doc_id))
    elif args.command == "ask":
        service = QAService(kb, settings)
        _print_json(
            service.ask(
                args.query,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                score_threshold=args.threshold,
                mode=args.mode,
            )
        )
    elif args.command == "agent":
        _print_json(run_agent(args.query, kb, settings))
    elif args.command == "tools":
        _print_json(get_tool_schemas())
    elif args.command == "call-tool":
        tool_args = _load_tool_args(args.args, args.args_file)
        executor = ToolExecutor(kb, settings)
        _print_json(executor.execute(args.name, tool_args))
    elif args.command == "stats":
        _print_json(kb.stats())
    elif args.command == "reset":
        kb.reset()
        _print_json({"ok": True})


if __name__ == "__main__":
    main()
