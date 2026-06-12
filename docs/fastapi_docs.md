# FastAPI Documentation

## Introduction

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python based on standard Python type hints.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go. One of the fastest Python frameworks available.
- **Fast to code**: Increase the speed to develop features by about 200% to 300%.
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors.
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging.
- **Easy**: Designed to be easy to use and learn.
- **Short**: Minimize code duplication.
- **Robust**: Get production-ready code with automatic interactive documentation.
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema.

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

## First Steps

### Create a simple application

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### Run the application

```bash
uvicorn main:app --reload
```

The `--reload` flag makes the server restart after code changes. Only use for development.

## Path Parameters

You can declare path parameters with the same syntax used by Python format strings:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

FastAPI automatically validates types. If `item_id` is not an integer, it returns a 422 Unprocessable Entity error.

### Predefined values with Enum

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    return {"model_name": model_name}
```

## Query Parameters

Parameters not declared as path parameters are automatically interpreted as query parameters:

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

URL: `/items/?skip=0&limit=10`

### Optional parameters

Use `Optional` from typing or just `= None`:

```python
from typing import Optional

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
```

## Request Body

Use Pydantic models to declare request bodies:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

## Response Model

You can declare the response model using the `response_model` parameter:

```python
@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item
```

## Status Codes

Use the `status_code` parameter:

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(name: str):
    return {"name": name}
```

## Dependency Injection

FastAPI provides a simple but powerful dependency injection system:

```python
from fastapi import Depends

async def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

## Error Handling

Raise HTTPException for HTTP errors:

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

## Middleware

Add middleware for cross-cutting concerns:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Background Tasks

FastAPI supports background tasks that run after returning a response:

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(message)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, f"Sending email to {email}")
    return {"message": "Notification sent"}
```

## File Uploads

Handle file uploads with UploadFile:

```python
from fastapi import File, UploadFile

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
```
