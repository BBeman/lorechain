from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from .ingest import get_retriever, MODEL_NAME
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

@tool("Hybrid_retriever", description="Hybrid Rag retriver for storyline lore questions")
def retrieved(query : str):
    docs = get_retriever().invoke(query)
    return "\n\n".join(d.page_content for d in docs)

#agents are lazy same as get_retriever in ingest, building them needs an api key
#so we only do it on first use, that way importing this file for tests doesnt crash
_architect = None
_keeper = None

def get_architect():
    global _architect
    if _architect is None:
        _architect = create_agent(
            model= f"openai:{MODEL_NAME}",
            tools = [retrieved],
            system_prompt="You are the Lore Architect, your role is to propose new lore on request, grounded in retrieved canon so it fits the world. You must not create Lore that does not match the current world state."
        )
    return _architect

def get_keeper():
    global _keeper
    if _keeper is None:
        _keeper = create_agent(
            model= f"openai:{MODEL_NAME}",
            tools = [retrieved],
            system_prompt="You are the continuity keeper,begin your reply with the single word CONSISTENT or CONTRADICTION, then explain your reasoning. you critique proposal against retrieved canon and flag contradictions of proposed new lore against retrieved canon and make sure it matches against the current world lore."
        )
    return _keeper

class State(TypedDict):
    request : str
    proposal : str
    is_consistent: bool
    feedback: str
    attempts: int


def Architect_node(state: State) -> dict:
    content = state["request"]
    if state["feedback"]:            
        content += f"\n\nYour previous proposal was rejected. Fix this: {state['feedback']}"
    result = get_architect().invoke({"messages": [{"role": "user", "content": content}]})
    return {"proposal": result["messages"][-1].content, "attempts": state["attempts"] + 1}


def parse_verdict(text: str) -> bool:
    # the keeper is told to begin its reply with CONSISTENT or CONTRADICTION.
    # we approve only when the reply starts with CONSISTENT, anything else is a rejection.
    return text.strip().upper().startswith("CONSISTENT")


def Keeper_node(state: State) -> dict:
    result = get_keeper().invoke({"messages": [{"role": "user", "content": state["proposal"]}]})
    text = result["messages"][-1].content
    return {"is_consistent": parse_verdict(text), "feedback": text}


def route(state):
    if state["attempts"] >=3:
        return "done"
    return "done" if state["is_consistent"] else "loop"

workflow = (
    StateGraph(State)
    .add_node("architect", Architect_node)
    .add_node("keeper", Keeper_node)
    .add_edge(START, "architect")
    .add_edge("architect","keeper")
    .add_conditional_edges("keeper", route, {"done": END, "loop": "architect"})
    .compile()
)
def create_lore(request: str):
    result = workflow.invoke({"request": request, "proposal": "", "is_consistent": False, "feedback": "", "attempts": 0})
    return f"final lore:\n{result['proposal']}"


