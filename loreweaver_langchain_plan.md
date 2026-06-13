# Loreweaver: a staged LangChain build plan

A multi-agent, hybrid-RAG worldbuilding engine. You feed it a "worldbook" (markdown files of fictional lore: characters, places, factions, events) and it becomes a council of AI agents that can answer questions about your world, invent new lore that stays consistent with canon, and catch its own contradictions.

It is deliberately not a CRUD app or a notes app or the usual "chat with your PDF" demo. It is a generative, self-checking creative system with a visible multi-agent feedback loop, which is what makes it stand out on a GitHub profile.

## Why this project teaches what you want

- **Multi-agent** is the whole point: an Architect proposes lore, a Continuity Keeper critiques it against canon, and a graph loops between them until the lore is consistent.
- **Hybrid RAG has a real motivation here**, not a bolted-on one. Lore is full of unique proper nouns (character and place names) where pure semantic search is weak, while thematic questions ("which factions distrust each other?") need semantic search. Keyword search (BM25) plus dense vector search covers both. You will literally see, in Stage 3, a query that dense-only gets wrong and hybrid gets right. That is the clearest way to understand why hybrid exists.

## How to use this plan

1. Do one stage per session. Each stage is a single git commit and push.
2. **Write the code yourself from the linked docs.** Do not paste a tutorial wholesale. The point is production fluency, not recognition. The docs are your reference, the build is your reps.
3. At the end of each stage, write a short entry in `STUDY_LOG.md` answering the "what you should now be able to explain" prompt in your own words. That log is both your retention mechanism and a genuine portfolio asset (it shows you understand the framework, not just that you wired it together).
4. Do not skip ahead. Each stage assumes the previous one works.

## A note on accuracy (verified against the live v1 docs and API reference)

LangChain v1 split the ecosystem into several packages. Knowing which package owns what saves you most import headaches:

- `langchain` is the lean core: `create_agent` (`langchain.agents`), the `tool` decorator (`langchain.tools`).
- `langchain_core` holds primitives like `InMemoryVectorStore` (`langchain_core.vectorstores`) and `Document`.
- `langchain_community` holds community integrations: document loaders (`langchain_community.document_loaders`) and `BM25Retriever` (`langchain_community.retrievers`).
- `langchain_text_splitters` is its own package for splitters (`RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`).
- `langchain_classic` holds stable-but-legacy utilities. Notably `EnsembleRetriever` lives here in v1: `from langchain_classic.retrievers import EnsembleRetriever`. It is still maintained (current as of the v1.3 line) and fine to use; the "classic" name just signals it predates the v1 core redesign.
- `langgraph` provides the orchestration layer (`langgraph.graph.StateGraph`).

All `create_agent`, `BM25Retriever`, `InMemoryVectorStore`, `RecursiveCharacterTextSplitter`, and `StateGraph` usages in this plan were confirmed against the current docs and API reference. If any import still fails after a version bump, the API reference (https://reference.langchain.com/python/) is the source of truth.

Key reference hubs (bookmark these):
- Docs home: https://docs.langchain.com/oss/python/langchain/overview
- API reference: https://reference.langchain.com/python/
- Providers (pick yours): https://docs.langchain.com/oss/python/integrations/providers/overview

## Pick your model provider first

The code is nearly identical across providers; only the model string and install line change.

- **AWS Bedrock** (closest to your day job): `pip install langchain langchain-aws`, then `create_agent(model="anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse", ...)`.
- **Ollama** (free, local, runs on your machine): `pip install langchain langchain-ollama`, then `create_agent(model="ollama:<your-local-model>", ...)`. Best for keeping cost at zero while learning.
- **OpenAI**: `pip install langchain "langchain[openai]"`, then `create_agent(model="openai:<model>", ...)`.

For embeddings (Stage 2 onward) you need an embeddings model from the same provider family (for example Bedrock embeddings via `langchain-aws`, or `OllamaEmbeddings` locally). The vector store and embeddings docs show the exact init per provider.

---

# Stage 0: Setup and first agent call

**Commit:** `chore: project setup and first LangChain agent call`

**Build:**
- Create the repo, a virtual environment, `requirements.txt`, a `.gitignore`, and a `.env` for credentials (never commit `.env`).
- Install LangChain plus your provider package.
- Write `hello.py` that constructs a `create_agent` (no custom tools yet) and invokes it with one message, then prints the reply.

The verified v1 pattern:
```python
from langchain.agents import create_agent

agent = create_agent(
    model="ollama:your-model",   # or your Bedrock / OpenAI string
    system_prompt="You are a helpful assistant.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "Say hello."}]})
print(result["messages"][-1].content_blocks)
```

**Docs:**
- Install: https://docs.langchain.com/oss/python/langchain/install
- Quickstart: https://docs.langchain.com/oss/python/langchain/quickstart
- Overview: https://docs.langchain.com/oss/python/langchain/overview

**Done when:** running `python hello.py` prints a real model response, and the repo is pushed to GitHub with `.env` excluded.

**What you should now be able to explain:** what `create_agent` is (model plus a harness of prompt, tools, and loop), the `{"messages": [...]}` invoke format, and how LangChain's standard interface lets you swap providers by changing one string.

---

# Stage 1: The worldbook, loaded and chunked

**Commit:** `feat: load and chunk the worldbook`

**Build:**
- Create a `worldbook/` folder with 4 to 6 markdown files of lore you invent (or borrow): a few characters, a couple of places, a faction or two, one or two events. Pack them with proper nouns on purpose, because Stage 3 depends on it.
- Write `ingest.py` that loads every markdown file with a loader from `langchain_community.document_loaders` (`TextLoader` or `DirectoryLoader`) and splits the text into chunks with `from langchain_text_splitters import RecursiveCharacterTextSplitter` (set `chunk_size` and `chunk_overlap`). Since your worldbook is markdown, `MarkdownHeaderTextSplitter` is a strong alternative that splits on headers and keeps each lore entry intact. Print the chunk count and one sample chunk including its `source` metadata.

**Docs:**
- Retrieval concepts (loaders, splitters, why chunking): https://docs.langchain.com/oss/python/langchain/retrieval
- Semantic search tutorial (loading and splitting in practice): https://docs.langchain.com/oss/python/langchain/knowledge-base

**Done when:** `python ingest.py` prints N chunks, each a `Document` carrying `page_content` and `source` metadata.

**What you should now be able to explain:** what a `Document` is, why you split text into chunks (context windows and search granularity), and what trade-off chunk size and overlap control.

---

# Stage 2: Dense vector RAG (Loremaster v1)

**Commit:** `feat: dense retrieval and grounded question answering`

**Build:**
- Embed your chunks and store them in a vector store. Use `InMemoryVectorStore` to start (no external database, ideal for learning); note Chroma or FAISS as drop-in alternatives for persistence.
- Turn the store into a retriever, then write a simple flow: take a question, retrieve the top-k chunks, stuff them into a prompt, and have the model answer using only that context. Print which chunks were used.

**Docs:**
- Build a searchable knowledge base (embeddings, vector store, retriever): https://docs.langchain.com/oss/python/langchain/knowledge-base
- Build a RAG workflow: https://docs.langchain.com/oss/python/langchain/rag
- Vector store integrations: https://docs.langchain.com/oss/python/integrations/vectorstores

**Done when:** you ask "who is [a character]?" and get an answer grounded in retrieved chunks, and you can print the exact chunks that produced it.

**What you should now be able to explain:** what an embedding is, what a vector store and a retriever do, and the two-step RAG pattern (retrieve, then generate). Note where pure semantic search already feels shaky on exact-name queries; that observation sets up the next stage.

---

# Stage 3: Hybrid RAG (the centerpiece)

**Commit:** `feat: hybrid retrieval with BM25 plus dense ensemble`

**Build:**
- Add a BM25 keyword retriever over the same chunks (`BM25Retriever`, needs `pip install rank_bm25`).
- Combine the BM25 retriever and your dense retriever with an `EnsembleRetriever`, which fuses their results using Reciprocal Rank Fusion. Start with `weights=[0.5, 0.5]`.
- Then prove the point: find a query with an exact proper noun where dense-only retrieval ranks the right chunk poorly, and show that hybrid surfaces it. Record the before/after in `STUDY_LOG.md`.

Verified imports for v1:
```python
from langchain_community.retrievers import BM25Retriever       # needs: pip install rank_bm25
from langchain_classic.retrievers import EnsembleRetriever     # moved to langchain_classic in v1

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5],   # tune these
    # c=60               # optional Reciprocal Rank Fusion constant
)
```
`EnsembleRetriever` fuses the two result lists with Reciprocal Rank Fusion (the `c` attribute is the RRF constant). The `weights` shift how much each retriever contributes.

**Docs:**
- Retriever integrations (BM25, Ensemble, exact imports): https://docs.langchain.com/oss/python/integrations/retrievers
- Retrieval concepts: https://docs.langchain.com/oss/python/langchain/retrieval
- API reference (to confirm any moved import): https://reference.langchain.com/python/

**Done when:** a proper-noun query returns the correct chunk through hybrid retrieval that dense-only ranked low or missed, and you can explain in your own words why, including what the `weights` do.

**What you should now be able to explain:** the difference between sparse (BM25, exact-term) and dense (embedding, semantic) retrieval, what Reciprocal Rank Fusion does, why hybrid beats either alone on a name-heavy corpus, and how the weights shift the balance.

---

# Stage 4: Loremaster as a real agent

**Commit:** `feat: Loremaster agent with hybrid retrieval as a tool`

**Build:**
- Wrap your hybrid retriever in a function decorated as a tool, then hand that tool to a `create_agent` Loremaster.
- Now the agent decides when to retrieve, instead of you hardcoding it. Test a lore question (it should choose to retrieve) and a non-lore question (it can choose not to). This is the docs' "agentic RAG" versus the "2-step RAG" you built in Stage 2.

**Docs:**
- Agents and tools: https://docs.langchain.com/oss/python/langchain/agents
- RAG (agentic vs two-step): https://docs.langchain.com/oss/python/langchain/rag

**Done when:** the Loremaster, given a lore question, autonomously calls the retrieval tool and answers from canon, and you can see the tool call in the result.

**What you should now be able to explain:** what a tool is, how the agent loop lets the model decide to call it, and the difference between agentic RAG (model chooses to retrieve) and static two-step RAG (you always retrieve).

---

# Stage 5: The council (multi-agent via LangGraph)

**Commit:** `feat: multi-agent worldbuilding workflow`

**Build:**
- Add an **Architect** agent that proposes new lore on request, grounded in retrieved canon so it fits the world.
- Add a **Continuity Keeper** agent that critiques the proposal against retrieved canon and flags contradictions.
- Orchestrate them in a LangGraph `StateGraph`: Architect proposes, Keeper checks, and a conditional edge loops back to the Architect with the Keeper's feedback if there is a contradiction, otherwise returns the approved lore. Give each agent a `name` so it slots in as a graph node.

**Docs:**
- Multi-agent custom workflow (StateGraph, nodes, conditional edges): https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- Agents (the `name` param for use as a subgraph node): https://docs.langchain.com/oss/python/langchain/agents

**Done when:** you request a new piece of lore, watch the Architect propose, the Keeper approve or reject, the graph loop on a rejection, and a consistency-checked result come back. You can trace how the state moved through the graph.

**What you should now be able to explain:** what a `StateGraph` and its `State` are, the difference between a normal edge and a conditional edge, how shared state passes information between agents, and how a single agent becomes one node in a larger multi-agent graph.

---

# Stage 6: Polish, demo, and portfolio framing

**Commit:** `docs: README, architecture diagram, demo, and study log`

**Build:**
- Write a real README: what Loreweaver is, an architecture diagram (a simple boxes-and-arrows of the council and the hybrid retriever is fine), setup and run instructions, and 2 or 3 example transcripts showing the loop catching a contradiction.
- Add a small entry point: a CLI is enough; a minimal Streamlit or Gradio UI is a nice bonus.
- Finalise `STUDY_LOG.md` with your per-stage reflections. This is the "learning layer" that signals you understand the framework.
- Optional but high-value: turn on LangSmith tracing so the multi-agent run is visible step by step, and screenshot it for the README.

**Docs:**
- LangSmith tracing quickstart (optional observability): https://docs.langchain.com/langsmith/trace-with-langchain

**Done when:** a stranger can clone the repo, follow the README, run a demo, and understand the architecture without asking you anything, and `STUDY_LOG.md` documents what each stage taught you.

**What you should now be able to explain:** how to present an AI system so the design is legible, what tracing and observability buy you, and why a learning log strengthens an open-source project.

---

## Where this leaves you

By the end you will have built, by hand, from the docs: a working LangChain v1 stack, dense and hybrid RAG with a demonstrated reason hybrid matters, a tool-using agent, and a multi-agent graph with a self-correcting loop. That is a genuinely distinctive portfolio piece and a real working knowledge of the framework, stage by stage, each one yours rather than copied.
