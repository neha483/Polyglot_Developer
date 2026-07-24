from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import lizard
import tiktoken
from dotenv import load_dotenv
from git import Repo
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("code-knowledge")
DEFAULT_REPO = "https://github.com/codejsha/spring-rest-sakila"
TEXT_EXTENSIONS = {
    ".java", ".kt", ".kts", ".xml", ".yaml", ".yml", ".properties",
    ".sql", ".md", ".adoc", ".json", ".toml", ".gradle",
}
IGNORED_DIRS = {
    ".git", ".gradle", ".idea", ".vscode", "build", "target", "out",
    "node_modules", "generated", "generated-sources",
}


class MethodKnowledge(BaseModel):
    name: str
    signature: str
    file: str
    start_line: int | None = None
    end_line: int | None = None
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    http_mapping: str | None = None
    complexity: int | None = None
    parameters: list[str] = Field(default_factory=list)
    return_type: str | None = None


class ChunkKnowledge(BaseModel):
    summary: str
    components: list[dict[str, Any]] = Field(default_factory=list)
    methods: list[MethodKnowledge] = Field(default_factory=list)
    noteworthy_aspects: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class ProjectSynthesis(BaseModel):
    purpose: str
    functionality: list[str]
    architecture: str
    data_flow: str
    noteworthy_aspects: list[str]
    risks_and_limitations: list[str]


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    language: str
    content: str
    sha256: str
    lines: int


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    file: str
    language: str
    start_line: int
    end_line: int
    text: str
    tokens: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clone_or_open(source: str, destination: Path | None = None) -> tuple[Path, bool]:
    candidate = Path(source).expanduser()
    if candidate.exists():
        return candidate.resolve(), False
    destination = destination or Path(tempfile.mkdtemp(prefix="code-knowledge-")) / "repo"
    LOGGER.info("Cloning %s", source)
    Repo.clone_from(source, destination, depth=1)
    return destination.resolve(), True


def should_include(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in TEXT_EXTENSIONS
        and not any(part in IGNORED_DIRS for part in path.parts)
        and path.stat().st_size <= 1_500_000
    )


def detect_language(path: Path) -> str:
    mapping = {
        ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
        ".sql": "sql", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
        ".properties": "properties", ".md": "markdown", ".adoc": "asciidoc",
        ".json": "json", ".toml": "toml", ".gradle": "gradle",
    }
    return mapping.get(path.suffix.lower(), "text")


def read_repository(root: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    for path in sorted(root.rglob("*")):
        if not should_include(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            LOGGER.warning("Skipping %s: %s", path, exc)
            continue
        rel = path.relative_to(root).as_posix()
        files.append(SourceFile(
            path=path,
            relative_path=rel,
            language=detect_language(path),
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            lines=content.count("\n") + 1,
        ))
    return files


def token_encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str) -> int:
    return len(token_encoder(model).encode(text))


def split_by_lines(source: SourceFile, model: str, max_tokens: int, overlap_lines: int = 12) -> list[CodeChunk]:
    lines = source.content.splitlines()
    chunks: list[CodeChunk] = []
    start = 0
    while start < len(lines):
        end = start
        current: list[str] = []
        while end < len(lines):
            candidate = "\n".join(current + [lines[end]])
            if current and count_tokens(candidate, model) > max_tokens:
                break
            current.append(lines[end])
            end += 1
        text = "\n".join(current)
        raw_id = f"{source.relative_path}:{start + 1}:{end}:{source.sha256}"
        chunks.append(CodeChunk(
            chunk_id=hashlib.sha1(raw_id.encode()).hexdigest()[:16],
            file=source.relative_path,
            language=source.language,
            start_line=start + 1,
            end_line=end,
            text=text,
            tokens=count_tokens(text, model),
        ))
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def build_chunks(files: Iterable[SourceFile], model: str, max_tokens: int) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for source in files:
        chunks.extend(split_by_lines(source, model, max_tokens))
    return chunks


JAVA_METHOD_RE = re.compile(
    r"(?P<annotations>(?:^[ \t]*@[\w.]+(?:\([^\n]*\))?[ \t]*\n)*)"
    r"^[ \t]*(?P<modifiers>(?:(?:public|protected|private|static|final|abstract|synchronized|native|default)\s+)*)"
    r"(?P<return>[\w<>,.?\[\] ]+)\s+(?P<name>[A-Za-z_$][\w$]*)\s*"
    r"\((?P<params>[^)]*)\)\s*(?:throws\s+[^\{]+)?\{",
    re.MULTILINE,
)


def deterministic_java_methods(source: SourceFile) -> list[dict[str, Any]]:
    if source.language != "java":
        return []
    results: list[dict[str, Any]] = []
    complexity_by_line: dict[int, int] = {}
    try:
        analysis = lizard.analyze_file(str(source.path))
        for fn in analysis.function_list:
            complexity_by_line[fn.start_line] = fn.cyclomatic_complexity
    except Exception as exc:  # lizard may reject generated/unusual syntax
        LOGGER.debug("Complexity parse failed for %s: %s", source.relative_path, exc)
    for match in JAVA_METHOD_RE.finditer(source.content):
        start_line = source.content[:match.start()].count("\n") + 1
        annotations = re.findall(r"@([\w.]+)(?:\(([^\n]*)\))?", match.group("annotations") or "")
        annotation_text = ["@" + name + (f"({args})" if args else "") for name, args in annotations]
        params = [p.strip() for p in match.group("params").split(",") if p.strip()]
        signature = " ".join((match.group("modifiers") + match.group("return") + " " + match.group("name")).split())
        signature += f"({match.group('params').strip()})"
        http_mapping = next((a for a in annotation_text if "Mapping" in a), None)
        results.append({
            "name": match.group("name"),
            "signature": signature,
            "file": source.relative_path,
            "start_line": start_line,
            "annotations": annotation_text,
            "http_mapping": http_mapping,
            "complexity": complexity_by_line.get(start_line),
            "parameters": params,
            "return_type": " ".join(match.group("return").split()),
        })
    return results


def repository_metrics(files: list[SourceFile]) -> dict[str, Any]:
    language_files = Counter(f.language for f in files)
    language_lines = Counter()
    methods: list[dict[str, Any]] = []
    for source in files:
        language_lines[source.language] += source.lines
        methods.extend(deterministic_java_methods(source))
    complexities = [m["complexity"] for m in methods if m.get("complexity") is not None]
    return {
        "file_count": len(files),
        "total_lines": sum(f.lines for f in files),
        "files_by_language": dict(language_files.most_common()),
        "lines_by_language": dict(language_lines.most_common()),
        "method_count": len(methods),
        "complexity": {
            "average_cyclomatic": round(sum(complexities) / len(complexities), 2) if complexities else None,
            "maximum_cyclomatic": max(complexities) if complexities else None,
            "high_complexity_method_count": sum(c > 10 for c in complexities),
        },
        "largest_files": [
            {"file": f.relative_path, "lines": f.lines}
            for f in sorted(files, key=lambda item: item.lines, reverse=True)[:15]
        ],
    }


def llm_client(model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature, max_retries=3, timeout=120)


def analyze_chunk(llm: ChatOpenAI, chunk: CodeChunk) -> ChunkKnowledge:
    structured = llm.with_structured_output(ChunkKnowledge, method="json_schema")
    prompt = f"""You are analyzing one bounded chunk from a software repository.
Return only facts grounded in the supplied code. Do not invent missing behavior.
For methods, preserve exact signatures and file/line locations when visible.
Describe business purpose, responsibilities, dependencies, HTTP mappings, persistence,
security, validation, error handling, tests, and noteworthy complexity only when supported.

FILE: {chunk.file}
LANGUAGE: {chunk.language}
LINES: {chunk.start_line}-{chunk.end_line}

```{chunk.language}
{chunk.text}
```
"""
    return structured.invoke(prompt)


def synthesize_project(llm: ChatOpenAI, repo_name: str, chunk_summaries: list[dict[str, Any]], metrics: dict[str, Any]) -> ProjectSynthesis:
    structured = llm.with_structured_output(ProjectSynthesis, method="json_schema")
    payload = json.dumps({"repository": repo_name, "metrics": metrics, "chunk_findings": chunk_summaries}, ensure_ascii=False)
    # A second bounded reduction avoids feeding every raw source token into one prompt.
    return structured.invoke(
        "Synthesize a conservative project-level understanding from the provided extracted findings. "
        "Resolve duplicates, preserve uncertainty, and do not infer unsupported behavior.\n\n" + payload
    )


def merge_methods(deterministic: list[dict[str, Any]], llm_findings: list[ChunkKnowledge]) -> list[dict[str, Any]]:
    descriptions: dict[tuple[str, str], MethodKnowledge] = {}
    for finding in llm_findings:
        for method in finding.methods:
            descriptions[(method.file, method.signature)] = method
    merged: list[dict[str, Any]] = []
    for method in deterministic:
        enriched = descriptions.get((method["file"], method["signature"]))
        item = dict(method)
        item["description"] = enriched.description if enriched else "Detected statically; run LLM mode for a grounded semantic description."
        item["responsibilities"] = enriched.responsibilities if enriched else []
        if enriched:
            item["end_line"] = enriched.end_line
        merged.append(item)
    return sorted(merged, key=lambda m: (m["file"], m.get("start_line") or 0))


def select_chunks(chunks: list[CodeChunk], max_chunks: int | None) -> list[CodeChunk]:
    # Prioritize production source and build/configuration, then tests and docs.
    def rank(c: CodeChunk) -> tuple[int, str, int]:
        p = c.file.lower()
        if "/main/" in p and c.language in {"java", "kotlin"}: priority = 0
        elif c.language in {"kotlin", "gradle", "yaml", "properties", "sql"}: priority = 1
        elif "/test/" in p: priority = 2
        else: priority = 3
        return priority, c.file, c.start_line
    ordered = sorted(chunks, key=rank)
    return ordered[:max_chunks] if max_chunks else ordered


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    root, temporary = clone_or_open(args.source, Path(args.clone_dir) if args.clone_dir else None)
    try:
        files = read_repository(root)
        metrics = repository_metrics(files)
        deterministic = [m for f in files for m in deterministic_java_methods(f)]
        chunks = select_chunks(build_chunks(files, args.model, args.chunk_tokens), args.max_chunks)
        llm_findings: list[ChunkKnowledge] = []
        errors: list[dict[str, str]] = []

        if not args.no_llm:
            llm = llm_client(args.model)
            for index, chunk in enumerate(chunks, 1):
                LOGGER.info("Analyzing chunk %s/%s: %s:%s-%s", index, len(chunks), chunk.file, chunk.start_line, chunk.end_line)
                try:
                    llm_findings.append(analyze_chunk(llm, chunk))
                except Exception as exc:
                    LOGGER.exception("Chunk failed")
                    errors.append({"chunk_id": chunk.chunk_id, "error": str(exc)})
            synthesis = synthesize_project(
                llm, Path(root).name,
                [f.model_dump(mode="json") for f in llm_findings], metrics,
            ).model_dump(mode="json")
        else:
            synthesis = {
                "purpose": "Static-only mode does not infer project purpose. Run without --no-llm for semantic synthesis.",
                "functionality": [], "architecture": "", "data_flow": "",
                "noteworthy_aspects": [], "risks_and_limitations": [],
            }

        output = {
            "schema_version": "1.0.0",
            "generated_at": utc_now(),
            "repository": {
                "source": args.source,
                "name": Path(root).name,
                "local_path": str(root),
                "commit": _git_commit(root),
            },
            "analysis_configuration": {
                "model": None if args.no_llm else args.model,
                "chunk_token_limit": args.chunk_tokens,
                "chunks_discovered": len(build_chunks(files, args.model, args.chunk_tokens)),
                "chunks_analyzed": len(chunks) if not args.no_llm else 0,
                "max_chunks": args.max_chunks,
                "strategy": "deterministic static extraction + token-bounded LLM map/reduce",
            },
            "overview": synthesis,
            "metrics": metrics,
            "components": _merge_component_findings(llm_findings),
            "methods": merge_methods(deterministic, llm_findings),
            "noteworthy_aspects": _unique(x for f in llm_findings for x in f.noteworthy_aspects),
            "risks": _unique(x for f in llm_findings for x in f.risks),
            "processing_errors": errors,
            "files": [
                {"path": f.relative_path, "language": f.language, "lines": f.lines, "sha256": f.sha256}
                for f in files
            ],
        }
        return output
    finally:
        if temporary and not args.keep_clone:
            shutil.rmtree(root.parent, ignore_errors=True)


def _git_commit(root: Path) -> str | None:
    try:
        return Repo(root).head.commit.hexsha
    except Exception:
        return None


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _merge_component_findings(findings: list[ChunkKnowledge]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for finding in findings:
        for component in finding.components:
            key = json.dumps(component, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                result.append(component)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured knowledge from a code repository.")
    parser.add_argument("--source", default=DEFAULT_REPO, help="Git URL or local repository path")
    parser.add_argument("--output", default="output/knowledge.json")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--chunk-tokens", type=int, default=6000)
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional cost-control cap")
    parser.add_argument("--clone-dir", default=None)
    parser.add_argument("--keep-clone", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="Run deterministic extraction only")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    if not args.no_llm and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless --no-llm is used")
    result = run_analysis(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %s", output_path.resolve())


if __name__ == "__main__":
    main()
