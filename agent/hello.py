from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5.5",
    system_prompt = "You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "ask what is up"}]}
)

print(result["messages"][-1].content_blocks)