from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: Optional[str] = Field(None, description="Session identifier, auto-generated if not provided")
    message: str = Field(..., description="User message to send to the agent")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    session_id: str
    message: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str


class MessageModel(BaseModel):
    """Message model for chat history."""
    role: str
    content: str
    timestamp: Optional[datetime] = None


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    session_id: str
    history: List[MessageModel]

