# LangChain Documentation

## Overview

LangChain is a framework for developing applications powered by language models. It provides a standard interface for chains, agents, and memory.

## Core Concepts

### LLMs and Chat Models

LangChain supports two types of models:
- **LLMs**: Take a string input and return a string output
- **Chat Models**: Take a list of messages and return a message

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("What is LangChain?")
print(response.content)
```

### Prompt Templates

Prompt templates help create structured prompts:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}"),
])

prompt = template.invoke({"user_input": "Tell me about Python"})
```

### Chains with LCEL (LangChain Expression Language)

LCEL uses the pipe `|` operator to compose chains:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "programming"})
```

## Document Loaders

LangChain provides many document loaders:

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

# Load text file
loader = TextLoader("data.txt")
docs = loader.load()

# Load PDF
loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()

# Load from web
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

## Text Splitters

Split documents into chunks for embedding:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)
```

## Embeddings

Generate vector embeddings from text:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("Hello world")
```

## Vector Stores

Store and search document embeddings:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
results = vectorstore.similarity_search("how to install", k=3)
```

### FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Load later
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

## Retrievers

Retrievers provide a standard interface for fetching documents:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

docs = retriever.invoke("how does retrieval work?")
```

### Contextual Compression Retriever

Filters out irrelevant parts of retrieved docs:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

## Memory

Add conversational memory to chains:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

## Agents

Agents use LLMs to decide which tools to use:

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What happened in AI news today?"})
```

## RAG (Retrieval Augmented Generation)

A complete RAG pipeline:

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "How do I use LangChain?"})
print(result["result"])
print(result["source_documents"])
```
