import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# 1. Ingestion: read PDF and create Document objects
reader = pypdf.PdfReader('data.pdf')
docs = [
    Document(
        page_content=page.extract_text() or "",
        metadata={"source": "data.pdf", "page": i}
    )
    for i, page in enumerate(reader.pages)
]

# 2. Chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 3. Embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=OLLAMA_BASE_URL)

#4. Store in Vector database
vector_store = Chroma.from_documents(
    chunks, embeddings, persist_directory='./rag_db'
)


#5. Retrieval
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)


#6. Augment Prompting with RAG
llm = OllamaLLM(model='llama3.1', temperature=0.5, base_url=OLLAMA_BASE_URL)

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n"
    "{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Retrieve first, so we keep a handle on the raw Documents
retrieved_docs = retriever.invoke("What is RAG?")

rag_chain = (
    {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#7. Generate answer and print sources
answer = rag_chain.invoke("What is RAG?")
print("Answer:\n", answer)
print("\nSources:")
for doc in retrieved_docs:
    page = doc.metadata['page']
    source = doc.metadata['source']
    snippet = doc.page_content[:100].replace("\n", " ").strip()
    print(f"  - {source}, page {page}: \"{snippet}...\"")