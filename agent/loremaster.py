from langchain.agents import create_agent
from langchain.tools import tool
from .ingest import get_retriever, MODEL_NAME
from  dotenv import load_dotenv
load_dotenv()

@tool("Hybrid_retriever", description="Hybrid Rag retriver for storyline lore questions")
def retrieved(query : str):
    docs = get_retriever().invoke(query)
    return "\n\n".join(d.page_content for d in docs)


agent = create_agent(
    model= f"openai:{MODEL_NAME}",
    tools = [retrieved],
    system_prompt = "You are the LoreMaster. Use the retriever tool to find lore and answer only from what it returns. for questions unrelated to swoletheim answer directly"
)

def ask_loremaster(question: str) -> str:
    result = agent.invoke(
        {"messages" : [{"role": "user", "content": question}]}
    )

    #Below is to just print the final answer
    return(result["messages"][-1].content)

    #We want to demonstrate tool calling
    #print(result["messages"])