from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

#loads every markdown file with loader
all_docs = []
for path in Path('worldbook').glob('*.md'):
    loader = TextLoader(path)
    all_docs.extend(loader.load())

print(len(all_docs))

#chunking md files with RCTS,(alternatively MD text splitter works)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
texts = text_splitter.split_documents(all_docs)

print(texts)
print(len(texts))
print(texts[0].metadata)
print(texts[0].page_content)
