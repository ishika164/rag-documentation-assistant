# LangGraph Documentation

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

LangGraph is inspired by Pregel and Apache Beam. The public interface draws inspiration from NetworkX.

## Core Concepts

### StateGraph

The main class for building workflows. You define:
1. A state schema (TypedDict or Pydantic)
2. Nodes (functions that transform the state)
3. Edges (connections between nodes, including conditional edges)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    messages: list
    counter: int

graph = StateGraph(MyState)
```

### Nodes

Nodes are Python functions that take the state and return an updated state:

```python
def my_node(state: MyState) -> MyState:
    # Do some work
    return {**state, "counter": state["counter"] + 1}

graph.add_node("my_node", my_node)
```

### Edges

**Simple edges** always go from node A to node B:

```python
graph.add_edge("node_a", "node_b")
```

**Conditional edges** route based on state:

```python
def decide_next(state: MyState) -> str:
    if state["counter"] > 5:
        return "end"
    return "continue"

graph.add_conditional_edges(
    "my_node",           # from this node
    decide_next,          # call this function
    {
        "end": END,       # if returns "end", go to END
        "continue": "my_node",  # if returns "continue", loop back
    }
)
```

### Entry Point and Compilation

```python
graph.set_entry_point("my_node")
app = graph.compile()
```

## Running the Graph

```python
result = app.invoke({"messages": [], "counter": 0})
```

## Checkpointing (Persistence)

LangGraph supports checkpointing for fault tolerance and human-in-the-loop:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
result = app.invoke(initial_state, config=config)
```

## Streaming

Stream node outputs as they complete:

```python
for event in app.stream(initial_state):
    for node_name, output in event.items():
        print(f"Node '{node_name}' completed")
        print(output)
```

## Multi-Agent Architectures

LangGraph supports complex multi-agent patterns:

### Supervisor Pattern

```python
# Supervisor decides which agent to call next
def supervisor_node(state):
    # LLM decides: agent_1, agent_2, or FINISH
    decision = llm.invoke(supervisor_prompt)
    return {"next": decision}

graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"agent_1": "agent_1", "agent_2": "agent_2", "FINISH": END}
)
```

### Parallel Execution

```python
# Run multiple nodes in parallel using Send API
from langgraph.constants import Send

def route_parallel(state):
    return [Send("worker", {"task": t}) for t in state["tasks"]]

graph.add_conditional_edges("dispatcher", route_parallel)
```

## Agentic RAG with LangGraph

The recommended pattern for self-corrective RAG:

```python
class RAGState(TypedDict):
    question: str
    documents: list
    generation: str
    web_search: bool

# Nodes
def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def grade_documents(state):
    filtered = []
    for doc in state["documents"]:
        score = grader_chain.invoke({
            "question": state["question"],
            "document": doc.page_content
        })
        if score.binary_score == "yes":
            filtered.append(doc)
    return {"documents": filtered, "web_search": len(filtered) == 0}

def generate(state):
    generation = rag_chain.invoke({
        "context": state["documents"],
        "question": state["question"]
    })
    return {"generation": generation}

# Routing
def decide_to_generate(state):
    if state["web_search"]:
        return "web_search"
    return "generate"

# Build graph
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search": "web_search_node", "generate": "generate"}
)
workflow.add_edge("generate", END)

app = workflow.compile()
```

## State Schema Design

Tips for designing state:
- Include all data that flows between nodes
- Use Optional types for fields that may not be populated early
- Track control flow variables (retry count, flags) in state
- Use Annotated with `operator.add` for fields that accumulate (like messages)

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]  # messages accumulate
    question: str
    retry_count: int
```
