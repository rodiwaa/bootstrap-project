# Agentic System Bootstrap

**Reference implementations for building production-ready agentic systems — one branch per component.**

> [BLog - https://rodiwa.substack.com/p/a-simple-ai-project-bootstrap](https://rodiwa.substack.com/p/a-simple-ai-project-bootstrap)

Starting an AI project from scratch every time is slow. This repo gives you a working, runnable reference for each major building block of an agentic system. Clone a branch, run the Makefile, and you're coding — not configuring.

## What's Here

Each branch is a self-contained implementation of one system component. Pick what you need, use it as a starting point, and build on top of it.

| Branch | What It Is |
|---|---|
| `main` | You're here — overview and navigation |
| `langgraph-chainlit-docker-nginx` | Full RAG stack: LangGraph backend + Chainlit UI + Docker + Nginx |
| `chainlit-interface` | Chainlit UI wired up to a LangGraph backend |
| `fastapi-interface` | FastAPI interface as an alternative to Chainlit |
| `multi-data-sources-rag` | RAG pipeline pulling from multiple data sources |
| `memory-prompt-caching` | Conversation memory + prompt caching strategies |
| `orchestration-a2a-mcp` | Multiple small agents with MCP tools, communicating via A2A protocol |
| `superviser-agent` | Supervisor/worker pattern — complex queries decomposed into subtasks, delegated to worker nodes (compare cost vs monolithic) |
| `deterministic-subagents` | Subagents with deterministic, rule-based routing |
| `threads` | Thread and session management across conversations |
| `hitl` | Human-in-the-loop: approval flows, email sending *(future)* |
| `realtime-ingestion` | Real-time data ingestion pipeline |
| `metrics` | Metrics collection and monitoring |
| `evals` | Evaluation framework setup |
| `edd` | Evaluation-driven design — golden datasets, rule-based + LLM-as-judge evals, runs on every commit via CI, feedback pushed to LangSmith |
| `e2e-evals-cicd` | End-to-end evals wired into CI/CD |
| `core-ai-mlops` | Core ML/AI ops patterns — experiment tracking, model management |
| `computer-vision` | Simple CV pipeline: image reading and classification *(future)* |
| `voice` | Voice interface integration |
| `sts` | Speech-to-speech using open-source STS models *(future)* |

## How To Use This Repo

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd bootstrap-project
   ```

2. **Checkout the branch you want**
   ```bash
   git checkout chainlit-interface
   # or whichever component you need
   ```

3. **Set up environment**
   ```bash
   cp .env-example .env
   # Fill in your API keys
   ```

4. **Install dependencies and run**
   ```bash
   make start-venv
   make start-server       # or start-chainlit, depending on branch
   ```

5. **For Docker deployment**
   ```bash
   make docker-compose-build
   make docker-compose-up
   ```
   Your app should be live at `http://localhost:8000`.

> Each branch has its own README with branch-specific setup details.

## Why This Exists

Most tutorials give you toy examples. Most production codebases are too complex to learn from. This sits in the middle — real patterns, minimal noise, easy to run.

The goal is to have a reference for every major component of an agentic system. When you're building something new, you shouldn't be searching Stack Overflow for boilerplate — you should be reading working code.

## Stack Choices

Branches generally use:

- **LangGraph** — agent orchestration
- **Chainlit or FastAPI** — user interface
- **Qdrant / in-memory** — vector storage
- **LangSmith / Opik** — observability and evals
- **uv** — dependency management
- **Docker + Nginx** — deployment

Not every branch uses all of these. Check the branch README for specifics.

## Project Structure (per branch)

```
src/
  graph/        # LangGraph pipeline and state
  nodes/        # Individual processing nodes
  utils/        # Shared utilities

main.py         # Entry point
Makefile        # Dev commands
docker-compose.yml
pyproject.toml
```

## Notes

- Branches marked ***(future)*** are scaffolded but not fully implemented yet.
- The `langgraph-chainlit-docker-nginx` branch is the most complete reference — start there if you're new.
- Uses absolute imports throughout. See [Python Modules & Packages](https://realpython.com/python-modules-packages/) if that causes issues.

## Final Thoughts

Use this as a starting point for your AI/ML engineering work. Fork it, strip out what you don't need, and build on what you do.

If you find any configs or tokens I shouldn't have left in — let me know. Karma's a B!
