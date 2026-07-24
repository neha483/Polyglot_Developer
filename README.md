# Sakila Code Knowledge Extractor

A token-aware repository analysis program built for the primary codebase:

`https://github.com/codejsha/spring-rest-sakila`

It reads a local checkout or clones a Git repository, extracts deterministic facts from source files, measures Java method complexity, sends bounded chunks to an LLM through LangChain, and writes a consistent machine-readable JSON knowledge document.

## Deliverables

- `src/code_knowledge/analyzer.py` — repository loading, filtering, token counting, chunking, static analysis, LLM extraction, map/reduce synthesis, and JSON generation.
- `main.py` — command-line entry point.
- `output/knowledge.json` — verified bootstrap JSON for the selected repository. Running the program replaces it with full method- and file-level analysis.
- `requirements.txt`, `pyproject.toml`, and `.env.example` — installation and configuration.

## Selected model

The default is `gpt-4.1-mini`, selected as a practical balance of code comprehension, structured-output reliability, context capacity, latency, and cost. Set `OPENAI_MODEL` or pass `--model` to use another compatible OpenAI model.

The model is invoked through `langchain-openai`. Pydantic schemas and LangChain structured output constrain each response to validated JSON instead of relying on free-form text parsing.

## Approach

### 1. Repository acquisition and efficient reading

The `--source` argument accepts either a Git URL or an existing local directory. Remote repositories are shallow-cloned with GitPython (`depth=1`). The reader:

- walks files once;
- skips build output, VCS metadata, IDE folders, generated folders, and dependencies;
- allows only relevant source, build, configuration, SQL, test, and documentation extensions;
- skips files larger than 1.5 MB;
- reads with UTF-8 replacement handling;
- records SHA-256 hashes, language, and line counts for traceability.

For repeatable or offline analysis, clone the repository yourself and pass its path:

```bash
git clone https://github.com/codejsha/spring-rest-sakila.git
python main.py --source ./spring-rest-sakila
```

### 2. Token-limit management

Raw repositories are never placed into one prompt. Each file is split by lines using `tiktoken`, with a default maximum of 6,000 input tokens and a 12-line overlap. Metadata identifies every chunk by file and line range.

The analysis uses a map/reduce design:

1. **Map:** each bounded code chunk is converted into validated `ChunkKnowledge` JSON.
2. **Deterministic merge:** exact Java method signatures, annotations, line locations, and complexity are extracted locally and joined with LLM descriptions.
3. **Reduce:** compact chunk findings—not raw source—are synthesized into the project overview.

This keeps prompt size predictable, avoids truncation, and preserves exact syntax through deterministic extraction. `--max-chunks` provides an explicit cost-control option while prioritizing production Java/Kotlin, build/configuration files, tests, and documentation in that order.

### 3. Static extraction and complexity

A deterministic Java scanner extracts:

- method names and signatures;
- modifiers, return types, and parameters;
- annotations and HTTP mapping annotations;
- source file and start line.

The `lizard` library calculates cyclomatic complexity per Java method. The final output includes average and maximum complexity, high-complexity method count, language/file statistics, and largest files.

The static scanner intentionally complements rather than replaces the LLM. Exact source facts come from parsing; semantic descriptions come from the model.

### 4. LLM knowledge extraction

Each chunk prompt explicitly prohibits unsupported inference. The model returns:

- a grounded summary;
- components and responsibilities;
- methods, signatures, descriptions, mappings, and dependencies;
- noteworthy design aspects;
- risks or limitations supported by the code.

The final project synthesis deduplicates findings and describes purpose, functionality, architecture, data flow, noteworthy aspects, and limitations.

## Installation

Requirements:

- Python 3.11+
- Git
- An OpenAI API key for semantic extraction

```bash
cd sakila-code-knowledge-extractor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## Run

Analyze the required GitHub repository:

```bash
python main.py \
  --source https://github.com/codejsha/spring-rest-sakila \
  --output output/knowledge.json
```

Analyze an existing checkout:

```bash
python main.py --source ../spring-rest-sakila --output output/knowledge.json
```

Use a different token budget or model:

```bash
python main.py --chunk-tokens 5000 --model gpt-4.1-mini
```

Cap the number of LLM calls during development:

```bash
python main.py --max-chunks 20
```

Run deterministic analysis without an API key:

```bash
python main.py --source ../spring-rest-sakila --no-llm
```

## Output schema

The generated JSON contains these top-level sections:

```json
{
  "schema_version": "1.0.0",
  "generated_at": "ISO-8601 timestamp",
  "repository": {},
  "analysis_configuration": {},
  "overview": {
    "purpose": "...",
    "functionality": [],
    "architecture": "...",
    "data_flow": "...",
    "noteworthy_aspects": [],
    "risks_and_limitations": []
  },
  "metrics": {},
  "components": [],
  "methods": [],
  "noteworthy_aspects": [],
  "risks": [],
  "processing_errors": [],
  "files": []
}
```

Every method includes its exact signature and source location when statically detectable. LLM failures are isolated by chunk and recorded under `processing_errors`; one failed request does not discard the rest of the analysis.

## Best practices used

- Shallow clone and single-pass filtered file loading.
- Token-aware chunks with overlap rather than character-only truncation.
- Deterministic extraction for exact facts; LLM extraction for semantic facts.
- Pydantic validation and model-native structured output.
- Map/reduce synthesis instead of submitting the full repository at once.
- Temperature zero and retries for stable extraction.
- SHA-256 file fingerprints and Git commit capture for traceability.
- Conservative prompts that prohibit invention.
- Partial-failure recording and reproducible JSON formatting.
- API keys loaded from environment variables rather than source code.

## Assumptions and limitations

- The repository is public or locally accessible. Private repositories require normal Git credentials outside this program.
- Java method extraction uses a lightweight scanner. It handles conventional Java declarations but is not a full compiler front end; deeply unusual syntax or generated code may require JavaParser or tree-sitter.
- Lizard complexity is measured only where its parser recognizes the source.
- LLM descriptions are grounded but probabilistic. Exact signatures, paths, hashes, and measured metrics remain deterministic.
- A full semantic run consumes API tokens and may incur cost. Use `--max-chunks` during iteration.
- The bundled `output/knowledge.json` is a verified bootstrap snapshot because the packaging environment could access GitHub pages but could not perform an outbound Git clone and had no user API key. Run the command above in a normal networked environment to produce the complete extracted JSON.

## Verified project baseline

The selected project is a Spring Boot REST API over MySQL's Sakila DVD-rental sample database. Its public documentation identifies Java 17, Gradle 7, MySQL 8, CRUD/report endpoints, Spring Data JPA, HATEOAS, Querydsl, MapStruct, REST Docs, and Actuator/Prometheus support. The build configuration also includes Spring Security, Redis support, JWT libraries, Blaze-Persistence, generated OpenAPI, and generated Postman artifacts.
