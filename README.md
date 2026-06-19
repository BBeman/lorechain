# Lorechain

A small worldbuilding engine for a fictional setting. You give it a folder of lore in markdown, and it does two things: answers questions about the world, and invents new lore that gets checked against the existing canon before it is accepted.

I built it stage by stage while learning LangChain v1 and LangGraph. The sample world is Swoletheim, a gym-fantasy setting where strength is magic and the forbidden supplement (`Trenbolune`) is basically steroids with a dramatic name.

## How it works

```
                         worldbook/  (markdown lore files)
                                 |
                                 v
                            ingest.py
            load  ->  chunk  ->  embed (OpenAI)  ->  InMemoryVectorStore
                                 |
                                 v
                 Hybrid Retriever  (EnsembleRetriever, RRF)
                   /                               \
        BM25 (keyword / exact term)        Dense (embeddings / semantic)
                   \                               /
                    \-------------+---------------/
                                  |
              +-------------------+----------------------+
              v                                          v
        loremaster.py                              council.py  (the Council)
        Agentic Q&A                        START -> Architect -> Keeper
        (agent decides                                 ^             |
         when to retrieve)                             |     conditional edge
                                                  loop (contradiction)
                                                   |             |
                                                       +-- approved ->+-> END
```

Three pieces:

- `ingest.py` loads the worldbook, chunks it, embeds it, and builds a hybrid retriever: BM25 keyword search fused with dense vector search via Reciprocal Rank Fusion. The hybrid setup is there because lore is full of unique proper nouns that semantic search alone handles poorly.
- `loremaster.py` is an agent that holds the retriever as a tool and decides whether a question needs it. Lore questions hit the retriever; "what is 2+2" gets answered directly.
- `council.py` is a LangGraph state machine. An Architect agent proposes new lore, a Continuity Keeper checks it against canon, and on a contradiction the graph loops back to the Architect with the feedback. It stops once the lore is consistent, or after three tries.

`lorechain_cli.py` ties them together with a menu: ask, or create.

## Stack

- LangChain v1, LangGraph
- OpenAI: `gpt-5.5` for chat, `text-embedding-3-small` for embeddings
- BM25 + `InMemoryVectorStore`, combined with `EnsembleRetriever`
- uv for packaging

## Run it

Needs [uv](https://docs.astral.sh/uv/) and an OpenAI key.

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run python -m agent.lorechain_cli
```

`agent/` is a package, so it runs as a module (`-m`) from the project root. To exercise the retriever on its own: `uv run python -m agent.ingest`.

## Examples

Asking the Loremaster. It calls the retriever, then answers from canon:

```
Ask the LoreMaster: who is vasquez Trenfell?

Vasquez Trenfell was once one of the greatest champions of the Iron Coliseum,
second only to Brock Ironhowe. At the Third Swolympiad he consumed Trenbolune.
His body swelled beyond mortal limits and then broke, an event remembered as the
Curse of Vasquez Trenfell, which triggered the Great Schism of Reps.
```

Ask it something off-topic ("what is 2+2") and it answers directly without retrieving.

The Council, asked to invent a faction. The Architect grounds it in canon, the Keeper approves it:

```
## The League of Measured Plates
A faction that rejects both sides of the Great Schism of Reps: they despise the
Enhanced Covenant for using mythical supplements, but mock the Order of the
Eternal Natural as naive zealots. Their creed: "What cannot be weighed, logged,
timed, or proven has no place beneath the bar."
```

## Notes

`STUDY_LOG.md` has my notes per stage, including the parts that did not go as expected. The clearest one: on a corpus this small, hybrid retrieval did not actually beat plain dense search, and the log works through why and when it would.
