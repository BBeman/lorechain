from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from ingest import build_retriever
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

retreiver = build_retriever()

@tool("Hybrid_retriever", description="Hybrid Rag retriver for storyline lore questions")
def retrieved(query : str):
    docs = retreiver.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

Architect = create_agent(
    model= "openai:gpt-5.5",
    tools = [retrieved],
    system_prompt="You are the Lore Architect, your role is to propose new lore on request, grounded in retrieved canon so it fits the world. You must not create Lore that does not match the current world state."
)

Continuity_keeper = create_agent(
    model= "openai:gpt-5.5",
    tools = [retrieved],
    system_prompt="You are the continuity keeper,begin your reply with the single word CONSISTENT or CONTRADICTION, then explain your reasoning. you critique proposal against retrieved canon and flag contradictions of proposed new lore against retrieved canon and make sure it matches against the current world lore."
)

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
    result = Architect.invoke({"messages": [{"role": "user", "content": content}]})
    return {"proposal": result["messages"][-1].content, "attempts": state["attempts"] + 1}


def Keeper_node(state: State) -> dict:
    result = Continuity_keeper.invoke({"messages": [{"role": "user", "content": state["proposal"]}]})
    text = result["messages"][-1].content
    is_consistent = text.strip().upper().startswith("CONSISTENT")
    return {"is_consistent": is_consistent, "feedback": text}


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

result = workflow.invoke({"request": "Invent a hero who reached the summit of Mount Swolympus before Brock Ironhowe", "proposal": "", "is_consistent": False, "feedback": "", "attempts": 0})
print("attempts:", result["attempts"])
print("approved:", result["is_consistent"])
print("final lore:\n", result["proposal"])


