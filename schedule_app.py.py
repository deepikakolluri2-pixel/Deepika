import os
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langserve import add_routes
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Add it in Render → Environment."
    )

# The project manages this fixed 30-day sample schedule.
START_DATE = datetime(2026, 8, 11)
END_DATE = datetime(2026, 9, 9)

CHROMA_PATH = "./chroma_schedule_db"
COLLECTION_NAME = "schedule_collection"


# ============================================================
# 2. SAMPLE 30-DAY SCHEDULE
# ============================================================

schedule_data = [
    {
        "id": "event_001",
        "date": "2026-08-12",
        "time": "10:00 AM",
        "title": "Team Meeting",
        "type": "Meeting",
        "description": "Weekly project discussion with the development team.",
    },
    {
        "id": "event_002",
        "date": "2026-08-13",
        "time": "2:00 PM",
        "title": "Python Workshop",
        "type": "Workshop",
        "description": "Hands-on workshop covering Python and data structures.",
    },
    {
        "id": "event_003",
        "date": "2026-08-14",
        "time": "11:00 AM",
        "title": "Complete Project Report",
        "type": "Task",
        "description": "Finish and submit the project documentation.",
    },
    {
        "id": "event_004",
        "date": "2026-08-15",
        "time": "3:00 PM",
        "title": "Project Discussion",
        "type": "Meeting",
        "description": "Discuss project progress and upcoming tasks.",
    },
    {
        "id": "event_005",
        "date": "2026-08-17",
        "time": "10:30 AM",
        "title": "Doctor Appointment",
        "type": "Appointment",
        "description": "Regular health check-up appointment.",
    },
    {
        "id": "event_006",
        "date": "2026-08-19",
        "time": "4:00 PM",
        "title": "Java Workshop",
        "type": "Workshop",
        "description": "Workshop on Java programming and object-oriented concepts.",
    },
    {
        "id": "event_007",
        "date": "2026-08-21",
        "time": "9:30 AM",
        "title": "Team Stand-up",
        "type": "Meeting",
        "description": "Project status and progress discussion.",
    },
    {
        "id": "event_008",
        "date": "2026-08-24",
        "time": "1:00 PM",
        "title": "Assignment Submission",
        "type": "Task",
        "description": "Submit the database management assignment.",
    },
    {
        "id": "event_009",
        "date": "2026-08-26",
        "time": "11:00 AM",
        "title": "Career Workshop",
        "type": "Workshop",
        "description": "Resume building and interview preparation.",
    },
    {
        "id": "event_010",
        "date": "2026-08-28",
        "time": "2:00 PM",
        "title": "Project Review Meeting",
        "type": "Meeting",
        "description": "Review the current project implementation.",
    },
    {
        "id": "event_011",
        "date": "2026-09-01",
        "time": "10:00 AM",
        "title": "Dentist Appointment",
        "type": "Appointment",
        "description": "Routine dental appointment.",
    },
    {
        "id": "event_012",
        "date": "2026-09-03",
        "time": "3:30 PM",
        "title": "Complete RAG Assignment",
        "type": "Task",
        "description": "Complete and test the Agentic RAG assignment.",
    },
    {
        "id": "event_013",
        "date": "2026-09-05",
        "time": "11:00 AM",
        "title": "AI Workshop",
        "type": "Workshop",
        "description": "Introduction to AI agents and retrieval augmented generation.",
    },
    {
        "id": "event_014",
        "date": "2026-09-07",
        "time": "2:30 PM",
        "title": "Final Project Meeting",
        "type": "Meeting",
        "description": "Final discussion about project completion.",
    },
    {
        "id": "event_015",
        "date": "2026-09-09",
        "time": "5:00 PM",
        "title": "Submit Project",
        "type": "Task",
        "description": "Submit the completed project.",
    },
]


# ============================================================
# 3. HELPERS
# ============================================================

def is_within_schedule_period(date_string: str) -> bool:
    """Return True when date is inside the 30-day project period."""
    try:
        date_value = datetime.strptime(date_string, "%Y-%m-%d")
        return START_DATE <= date_value <= END_DATE
    except ValueError:
        return False


def event_to_text(event: dict) -> str:
    return (
        f"Event ID: {event['id']}\n"
        f"Date: {event['date']}\n"
        f"Time: {event['time']}\n"
        f"Title: {event['title']}\n"
        f"Type: {event['type']}\n"
        f"Description: {event['description']}"
    )


# ============================================================
# 4. GEMINI EMBEDDINGS + CHROMADB
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# Rebuild the sample collection at startup so Render restarts do not
# accumulate duplicate schedule documents.
try:
    vector_store.delete_collection()
except Exception:
    pass

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

documents = [
    Document(
        page_content=event_to_text(event),
        metadata={
            "id": event["id"],
            "date": event["date"],
            "time": event["time"],
            "title": event["title"],
            "type": event["type"],
        },
    )
    for event in schedule_data
]

vector_store.add_documents(
    documents=documents,
    ids=[event["id"] for event in schedule_data],
)


# ============================================================
# 5. TOOL 1 — GET SCHEDULE
# ============================================================

@tool(response_format="content_and_artifact")
def get_schedule(query: str):
    """
    Retrieve relevant schedule information based on date, time,
    availability, event type, event title, or user query.
    """
    retrieved_docs = vector_store.similarity_search(query, k=5)

    if not retrieved_docs:
        return "No matching schedule entries found.", []

    serialized = "\n\n".join(
        f"Schedule Entry:\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


# ============================================================
# 6. TOOL 2 — UPDATE SCHEDULE
# ============================================================

@tool
def update_schedule(
    action: str,
    event_id: str = "",
    date: str = "",
    time: str = "",
    title: str = "",
    event_type: str = "",
    description: str = "",
):
    """
    Add, update, or remove a schedule event.

    action must be one of:
    add, update, remove
    """
    global schedule_data, vector_store

    if action not in {"add", "update", "remove"}:
        return "Invalid action. Use add, update, or remove."

    # -------------------------
    # ADD
    # -------------------------
    if action == "add":
        if not date:
            return "A date is required when adding an event."

        if not is_within_schedule_period(date):
            return (
                f"Event date must be between "
                f"{START_DATE.strftime('%Y-%m-%d')} and "
                f"{END_DATE.strftime('%Y-%m-%d')}."
            )

        existing_ids = {event["id"] for event in schedule_data}

        number = len(schedule_data) + 1
        new_id = f"event_{number:03d}"

        while new_id in existing_ids:
            number += 1
            new_id = f"event_{number:03d}"

        new_event = {
            "id": new_id,
            "date": date,
            "time": time,
            "title": title or "Untitled Event",
            "type": event_type or "Task",
            "description": description,
        }

        schedule_data.append(new_event)

        document = Document(
            page_content=event_to_text(new_event),
            metadata={
                "id": new_event["id"],
                "date": new_event["date"],
                "time": new_event["time"],
                "title": new_event["title"],
                "type": new_event["type"],
            },
        )

        vector_store.add_documents(
            documents=[document],
            ids=[new_id],
        )

        return f"Event added successfully: {new_event}"

    # -------------------------
    # UPDATE
    # -------------------------
    if action == "update":
        if not event_id:
            return "event_id is required when updating an event."

        for event in schedule_data:
            if event["id"] == event_id:

                if date:
                    if not is_within_schedule_period(date):
                        return (
                            f"Updated date must be between "
                            f"{START_DATE.strftime('%Y-%m-%d')} and "
                            f"{END_DATE.strftime('%Y-%m-%d')}."
                        )
                    event["date"] = date

                if time:
                    event["time"] = time

                if title:
                    event["title"] = title

                if event_type:
                    event["type"] = event_type

                if description:
                    event["description"] = description

                updated_document = Document(
                    page_content=event_to_text(event),
                    metadata={
                        "id": event["id"],
                        "date": event["date"],
                        "time": event["time"],
                        "title": event["title"],
                        "type": event["type"],
                    },
                )

                vector_store.delete(ids=[event_id])

                vector_store.add_documents(
                    documents=[updated_document],
                    ids=[event_id],
                )

                return f"Event updated successfully: {event}"

        return f"No event found with ID {event_id}"

    # -------------------------
    # REMOVE
    # -------------------------
    if action == "remove":
        if not event_id:
            return "event_id is required when removing an event."

        for event in schedule_data:
            if event["id"] == event_id:

                schedule_data.remove(event)
                vector_store.delete(ids=[event_id])

                return f"Event removed successfully: {event}"

        return f"No event found with ID {event_id}"


# ============================================================
# 7. GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# 8. AGENT — EXACTLY TWO TOOLS
# ============================================================

tools = [get_schedule, update_schedule]

system_prompt = """
You are an Agentic RAG-based Schedule Assistant.

Your ONLY responsibility is managing and answering questions
about the user's schedule.

The schedule period is:
August 11, 2026 to September 9, 2026.

You have exactly TWO tools:

1. get_schedule
   - Retrieves relevant schedule information.

2. update_schedule
   - Adds, updates, or removes schedule events.

STRICT RULES:

1. Use get_schedule for schedule questions.

2. Before updating or removing an existing event,
   use get_schedule first.

3. Use get_schedule to check availability and conflicts.

4. ONLY answer questions related to the user's schedule.

5. If a question is unrelated to the schedule,
   DO NOT answer it using general knowledge.

6. For unrelated questions, respond exactly:
   "I can only help with your schedule."

7. Do not invent schedule information.

8. Only manage events between August 11, 2026
   and September 9, 2026.

9. For adding an event, use update_schedule with action='add'.

10. For updating an event, use update_schedule with action='update'.

11. For removing an event, use update_schedule with action='remove'.

12. When the user says "tomorrow", use August 11, 2026
    as the reference date.

13. After a successful modification, clearly confirm the change.
"""

schedule_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)


# ============================================================
# 9. LANGSERVE INPUT / OUTPUT ADAPTER
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(description="Your schedule-related message")


def format_for_agent(value):
    user_input = value["input"] if isinstance(value, dict) else value.input
    return {
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    }


def extract_text_response(agent_output) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if not messages:
        return "No response generated."

    # Find the final AI message.
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue

        content = getattr(message, "content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)

            if text_parts:
                return "\n".join(text_parts)

        return str(content)

    return "No final response generated."


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | schedule_agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str,
)


# ============================================================
# 10. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    description=(
        "A 30-day Agentic RAG Schedule Assistant using "
        "Gemini, ChromaDB, LangChain and two tools."
    ),
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agentic RAG Schedule Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 850px;
                margin: 60px auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 { color: #7b1e3b; }
            .box {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 10px;
            }
            a { color: #7b1e3b; }
        </style>
    </head>
    <body>
        <h1>📅 Agentic RAG Schedule Assistant</h1>

        <div class="box">
            <p>
                Manage the user's schedule from
                <b>August 11, 2026</b> to
                <b>September 9, 2026</b>.
            </p>

            <p><b>Available operations:</b></p>
            <ul>
                <li>Retrieve schedule information</li>
                <li>Check availability</li>
                <li>Add events</li>
                <li>Update events</li>
                <li>Remove events</li>
            </ul>

            <p>
                <b>Agent tools:</b>
                get_schedule and update_schedule
            </p>

            <p>
                Open the
                <a href="/agent/playground/">
                    Schedule Assistant Playground
                </a>
                to interact with the agent.
            </p>

            <p>
                API documentation:
                <a href="/docs">/docs</a>
            </p>
        </div>
    </body>
    </html>
    """


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Agentic RAG Schedule Assistant",
        "tools": ["get_schedule", "update_schedule"],
        "schedule_period": "2026-08-11 to 2026-09-09",
    }


# ============================================================
# 11. RENDER START COMMAND
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
