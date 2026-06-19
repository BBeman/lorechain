from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# single source of truth for the chat model (used by loremaster + council)
MODEL_NAME = "gpt-5.5"

def build_retriever():

    #loads every markdown file with loader
    all_docs = []
    for path in Path('worldbook').glob('*.md'):
        loader = TextLoader(path)
        all_docs.extend(loader.load())

    #print(len(all_docs))

    #chunking md files with RCTS,(alternatively MD text splitter works)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(all_docs)

    #print(texts)
    #print(len(texts))
    #print(texts[0].metadata)
    #print(texts[0].page_content)

    #creating vector store using in memory, embedding using openai.
    vector_store = InMemoryVectorStore(embedding= OpenAIEmbeddings(model="text-embedding-3-small"))
    vector_store.add_documents(documents=texts)

    #turn vector store into a retriever
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    #results = dense_retriever.invoke(question)

    #BM25 for key word matching
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 3

    #ensemble retriever combines retrievals
    ensemble = EnsembleRetriever(retrievers= [bm25_retriever,dense_retriever],
    weights = [0.5,0.5]
    )

    return ensemble


# lazily-built, process-wide single retriever.
# build_retriever() ingests + embeds the whole worldbook, so we only ever
# want to do it once per run. Both loremaster and council import this, and
# because they share the same module the cache is shared across them too.
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


#direct test llm call
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    retriever = build_retriever()
    question = "Sacred Salt"
    results = retriever.invoke(question)


    model=ChatOpenAI(model = MODEL_NAME)
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
