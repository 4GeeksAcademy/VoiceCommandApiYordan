"""
VoiceCommandApiYordan - FastAPI application with Groq-powered voice command processing.
"""

import os
import json
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. Load environment variables & configure Groq client
# ---------------------------------------------------------------------------
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------------
# 2. CORS configuration
# ---------------------------------------------------------------------------
app = FastAPI(title="VoiceCommandApiYordan")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 3. In-memory task storage
# ---------------------------------------------------------------------------
tasks: list[dict] = []
_next_id: int = 1


def _next_task_id() -> int:
    global _next_id
    current = _next_id
    _next_id += 1
    return current


# ---------------------------------------------------------------------------
# 4. Pydantic models
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    title: str
    done: bool = False


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    done: Optional[bool] = None


class TaskResponse(BaseModel):
    id: int
    title: str
    done: bool


class InstructionRequest(BaseModel):
    transcription: str


# ---------------------------------------------------------------------------
# 5. CRUD endpoints for /tasks
# ---------------------------------------------------------------------------

@app.get("/tasks", response_model=list[TaskResponse])
def get_tasks():
    """Return the full list of tasks."""
    return tasks


@app.post("/tasks", response_model=TaskResponse, status_code=201)
def create_task(payload: TaskCreate):
    """Create a new task with an auto-incremented id."""
    task = {
        "id": _next_task_id(),
        "title": payload.title,
        "done": payload.done,
    }
    tasks.append(task)
    return task


@app.put("/tasks/{task_id}", response_model=TaskResponse)
def replace_task(task_id: int, payload: TaskCreate):
    """Completely replace a task (title and done)."""
    for task in tasks:
        if task["id"] == task_id:
            task["title"] = payload.title
            task["done"] = payload.done
            return task
    raise HTTPException(status_code=404, detail="Task not found")


@app.patch("/tasks/{task_id}", response_model=TaskResponse)
def partial_update_task(task_id: int, payload: TaskUpdate):
    """Partially update a task's title and/or done fields."""
    for task in tasks:
        if task["id"] == task_id:
            if payload.title is not None:
                task["title"] = payload.title
            if payload.done is not None:
                task["done"] = payload.done
            return task
    raise HTTPException(status_code=404, detail="Task not found")


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    """Delete a task and return a confirmation message."""
    for i, task in enumerate(tasks):
        if task["id"] == task_id:
            del tasks[i]
            return {"message": f"Task with id {task_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Task not found")


# ---------------------------------------------------------------------------
# 6. POST /instruction — Groq-powered voice command interpreter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a strict JSON command extractor. Your sole task is to parse the user's natural-language request "
    "and respond with a single, flat JSON object. You MUST NOT include any explanation, markdown formatting, "
    "code fences, or any text outside the JSON object.\n\n"
    "The JSON object MUST have exactly these three keys:\n"
    "  - \"endpoint\": a string — the API path\n"
    "  - \"method\": a string — the HTTP method\n"
    "  - \"params\": an object (may be empty {}) — the parameters for the operation\n\n"
    "Mapping rules (apply these strictly):\n\n"
    "1. LIST all tasks — if the user wants to see, list, show, or get all tasks:\n"
    "   {\"endpoint\": \"/tasks\", \"method\": \"GET\", \"params\": {}}\n\n"
    "2. CREATE a task — if the user wants to add, create, put a new item, or similar:\n"
    "   {\"endpoint\": \"/tasks\", \"method\": \"POST\", \"params\": {\"title\": \"<extracted title>\"}}\n\n"
    "3. PARTIALLY UPDATE a task (mark done/undone, change title) — if the user wants to mark, complete, "
    "rename, change, edit, update (partial) a specific task:\n"
    "   {\"endpoint\": \"/tasks/<id>\", \"method\": \"PATCH\", \"params\": {\"done\": true}}"
    "   or {\"endpoint\": \"/tasks/<id>\", \"method\": \"PATCH\", \"params\": {\"title\": \"<new title>\"}}"
    "   or both keys in params if the user provides both title and done.\n\n"
    "4. REPLACE a task completely — if the user says replace, overwrite, or sets both title and done explicitly:\n"
    "   {\"endpoint\": \"/tasks/<id>\", \"method\": \"PUT\", \"params\": {\"title\": \"...\", \"done\": <bool>}}\n\n"
    "5. DELETE a task — if the user wants to delete, remove, erase a specific task:\n"
    "   {\"endpoint\": \"/tasks/<id>\", \"method\": \"DELETE\", \"params\": {}}\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "- Extract the task ID from phrases like \"task 1\", \"task #1\", \"the first task\", \"id 1\", etc.\n"
    "- The \"title\" in params must be a clean string of the task description.\n"
    "- \"done\" in params must be a boolean true or false, never a string.\n"
    "- The \"endpoint\" must include the task ID number when the operation targets a specific task "
    "(e.g., \"/tasks/1\", \"/tasks/2\").\n"
    "- For GET (list) and POST (create), the endpoint is just \"/tasks\" (no trailing ID).\n"
    "- Never add extra keys to the JSON. Never wrap in markdown. Never add commentary.\n"
    "- Return ONLY the raw JSON object and nothing else."
)


@app.post("/instruction")
def process_instruction(payload: InstructionRequest):
    """
    Accept a natural-language instruction, send it to Groq's LLM,
    and return the structured JSON command parsed by the model.
    """
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload.transcription},
        ],
    )

    raw = chat_completion.choices[0].message.content
    # The LLM is instructed to return a flat JSON object; parse and return it.
    return json.loads(raw)