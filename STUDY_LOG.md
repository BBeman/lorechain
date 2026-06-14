This is just my log for learning langchain.

Stage 0:
At this stage all I have done is setup project and do a simple langchain agent call using openai api.
all it does is return "what's up?".

steps were installing langchain packages including langchain's openai sdk wrapper. i installed those using uv. i did not install deepagents as it's not needed and opted for dotenv which loads directly to the os from env file where the openai sdk will pick up the env. langchain docs show api setup using export but it means each terminal needs me to rewrite.

simple hello.py file for now imports langchain agents with create agent, uses model gpt 5.5 from openai with simple prompt. then attach to result an agent invokation of messages key with list role key use and content key asking it to just ask us what is up. then we get the full conversation grab the final message which is the model's answer and print its content as type blocks.


stage 1:
Worldbook folder has 5 markdown files. I used AI to generate these so I can focus on writing the code, they are simple bodybuilding related meme/lore. 

I added ingest.py . what is in there is just a loader using the langchain community import for loader. Using Path I fetched the path of the worldbook and looped through it loading the documents into a list. Then using langchain text splitters I used recursive character text splitter to chunk the documents in my list. I defined a chunk size of 500 and an overlap of 50, this gives us 19 chunks from 23 sections because it bundled short entries. Alternatively i could have used markdown text splitter which is exactly intended for our use case here, we wouldn't lose the header metadata for example. But for this project I wanted to do it this way.