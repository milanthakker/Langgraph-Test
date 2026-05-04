import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from phoenix.otel import using_session, using_user
from pydantic import BaseModel

from agent import build_agent
from database import get_itinerary, init_db

init_db()

agent = build_agent()

app = FastAPI(
    title="LangGraph Agent API",
    description="A simple API for interacting with a LangGraph agent that can search the web",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"
    with using_session(session_id=session_id), using_user(user_id):
        result = agent.invoke({"messages": [HumanMessage(content=request.message)]})
    return ChatResponse(response=result["messages"][-1].content, session_id=session_id)


@app.get("/itinerary/{itinerary_id}")
def get_itinerary_by_id(itinerary_id: int):
    """Retrieve a stored itinerary by ID."""
    itinerary = get_itinerary(itinerary_id)
    if not itinerary:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return itinerary


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
