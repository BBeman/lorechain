from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

#loads every markdown file with loader
all_docs = []
for path in Path('worldbook').glob('*.md'):
    loader = TextLoader(path)
    all_docs.extend(loader.load())

print(len(all_docs))

#chunking md files with RCTS,(alternatively MD text splitter works)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
texts = text_splitter.split_documents(all_docs)

#print(texts)
#print(len(texts))
#print(texts[0].metadata)
#print(texts[0].page_content)

question = "who is vasquez Trenfell?"

#creating vector store using in memory, embedding using openai.
vector_store = InMemoryVectorStore(embedding= OpenAIEmbeddings(model="text-embedding-3-small"))
vector_store.add_documents(documents=texts)

#turn vector store into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke(question)


#for now a direct llm call for our RAG
from langchain_openai import ChatOpenAI


model=ChatOpenAI(model = "gpt-5.5")

context = " ".join(doc.page_content for doc in results)
prompt = f'use this context to answer question: {question} : context : {context}'


messages = [
    {"role": "system", "content": "You are a lore master, Answer using only the provided context. If it is not in the context, say you do not know."},
    {"role": "user", "content": prompt},
]

response = model.invoke(messages)
print(response.content)

print("--- chunks used ---")
for doc in results:
    print(doc.metadata)
    print(doc.page_content)
