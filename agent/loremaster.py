from langchain.agents import create_agent
from langchain.tools import tool
from ingest import build_retriever
from  dotenv import load_dotenv
load_dotenv()

retreiver = build_retriever()

user_question = input("Ask the LoreMaster: ")

@tool("Hybrid_retriever", description="Hybrid Rag retriver for storyline lore questions")
def retrieved(query : str):
    docs = retreiver.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


agent = create_agent(
    model= "openai:gpt-5.5",
    tools = [retrieved],
    system_prompt = "You are the LoreMaster. Use the retriever tool to find lore and answer only from what it returns. for questions unrelated to swoletheim answer directly"
)

result = agent.invoke(
    {"messages" : [{"role": "user", "content": user_question}]}
)

#Below is to just print the final answer
#print(result["messages"][-1].content_blocks)

#We want to demonstrate tool calling
print(result["messages"])