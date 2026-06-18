This is just my log for learning langchain.

## Stage 0:

At this stage all I have done is setup project and do a simple langchain agent call using openai api.
all it does is return "what's up?".

steps were installing langchain packages including langchain's openai sdk wrapper. i installed those using uv. i did not install deepagents as it's not needed and opted for dotenv which loads directly to the os from env file where the openai sdk will pick up the env. langchain docs show api setup using export but it means each terminal needs me to rewrite.

simple hello.py file for now imports langchain agents with create agent, uses model gpt 5.5 from openai with simple prompt. then attach to result an agent invokation of messages key with list role key use and content key asking it to just ask us what is up. then we get the full conversation grab the final message which is the model's answer and print its content as type blocks.


## stage 1:

Worldbook folder has 5 markdown files. I used AI to generate these so I can focus on writing the code, they are simple bodybuilding related meme/lore. 

I added ingest.py . what is in there is just a loader using the langchain community import for loader. Using Path I fetched the path of the worldbook and looped through it loading the documents into a list. Then using langchain text splitters I used recursive character text splitter to chunk the documents in my list. I defined a chunk size of 500 and an overlap of 50, this gives us 19 chunks from 23 sections because it bundled short entries. Alternatively i could have used markdown text splitter which is exactly intended for our use case here, we wouldn't lose the header metadata for example. But for this project I wanted to do it this way.

## stage 2:
I embeded my chunks and stored them in a vector store, by embedding them using openai's embedding model. In this case an InMemoryVectorStore. being an in memory means that I had to have evefrything in the ingest.py file and I can't just seperate query against retrieval in different file structures until I implement a store with persistence not one that breaks down post run. Then I turned that store into a retriever which allows me to return documents given an unstructured query, then took the top 3 documents stuffing them into a prompt where I joined the query against the retrieved documents. For now using a direct llm call.

## stage 3:

I implemented BM25 key word matching to turn my dense RAG into a hybrid RAG, although knowing this was overkill I did it anyways for the sake of testing it out and naturally the corpus was too small to show its benefit, dense RAG was consistently ranking the correct chunks across multiple queries, I even tried to hardcode my question to just one made up word and still semantically they are all connected enough for the dense RAG to work perfectly. BM25 addition ontop did naturally retrieve more chunks and had typically more detailed responses but it was nothing that was necessary. When would this hybrid rag system really matter? It would matter if our corpus was larger had less semantic connectivity and very specific key wording, generally in systems where there are specific IDs needed matching. 

As for implementation we had our dense rag then we simply add a bm25 retriever and combine them using an ensemble retriever, the key thing is that bm25 scores and dense similarity scores are scaled differently , the ensemble ignores the raw scores and uses the document rank in each list, so high ranking in either floats to the top.And we added weights to the ensemble, we made them equal to make each retrivers contribution equal as it multiplies against their rank. for the sake of retrieval symetry i made both top_k from dense and bm25 equal.

## stage 4

Here I wrapped my Hybrid retriever in a tool decorator function, so that I could pass it over to an ai agent to reason it's usage. In our previous implementation we did a direct LLM call, I put that in an if main statement so it doesn't conflict with our ai agent but we had to gather context by looping through the content inside an invokation of our retriever and hard code it into the prompt.

 On the other hand I created a loremaster agent where I imported a function of our retriever then another function where we loop through our invoked retriever and returned the content. We wrapped that into a tool decorator called Hybrid_retriever and added a description. Then I simply added it as a tool into the agent. Our Agent now has to decide when to run the retriever function. the tool description paired with system_prompt is the main differentiator to this. I did a first stage test where I added into the description to always use it to answer questions, same in the system prompt. This gave us good answers when relevant to the lore but when we ask it a general question like 2+2 it gave us useless lore and did a tool call when it didn't need to. So the description needs to highlight when it is to be called and the system prompt requires the same and grounding language for say Lore related queries.

This time I called the whole messages output rather than the last content blocks because I wanted to see if the tool was being called, why it was called, when reasoning was used and so on. This is agentic Rag where an Agent decides when to use the RAG against query, we have tested both agent rag and static two step rag. Static is probably better for systems where you want retrieval every single time.