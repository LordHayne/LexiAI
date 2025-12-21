"""
Request models for the Lexi API.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import re

class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.
    """
    message: str = Field(..., description="The user message to process")
    user_id: str = Field(default="default", description="Unique identifier for the user")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation tracking")
    stream: bool = Field(False, description="Whether to stream the response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the chat")
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @validator('user_id')
    def user_id_valid(cls, v):
        if not v or not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError('User ID must be alphanumeric with optional underscores, hyphens, or dots')
        return v

class MemoryStoreRequest(BaseModel):
    """
    Request model for storing a memory.
    """
    content: str = Field(..., description="The content to store in memory")
    user_id: str = Field(..., description="Unique identifier for the user")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the memory")
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class BatchMemoryItem(BaseModel):
    """
    A single memory item in a batch operation.
    """
    content: str = Field(..., description="The content to store in memory")
    user_id: str = Field(..., description="Unique identifier for the user")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the memory")
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class BatchMemoryStoreRequest(BaseModel):
    """
    Request model for batch memory storage.
    """
    memories: List[BatchMemoryItem] = Field(..., description="List of memories to store")
    
    @validator('memories')
    def validate_memories(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one memory is required')
        if len(v) > 100:
            raise ValueError('Maximum batch size is 100 memories')
        return v

class BatchMemoryDeleteRequest(BaseModel):
    """
    Request model for batch memory deletion.
    """
    memory_ids: List[str] = Field(..., description="List of memory IDs to delete")
    user_id: str = Field(..., description="Unique identifier for the user")
    
    @validator('memory_ids')
    def validate_memory_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one memory ID is required')
        if len(v) > 100:
            raise ValueError('Maximum batch size is 100 memory IDs')
        return v

class MemoryUpdateItem(BaseModel):
    """
    A single memory update in a batch operation.
    """
    id: str = Field(..., description="ID of the memory to update")
    content: Optional[str] = Field(None, description="New content (if updating)")
    tags: Optional[List[str]] = Field(None, description="New tags (if updating)")
    category: Optional[str] = Field(None, description="New category (if updating)")

class BatchMemoryUpdateRequest(BaseModel):
    """
    Request model for batch memory updates.
    """
    updates: List[MemoryUpdateItem] = Field(..., description="List of memory updates")
    user_id: str = Field(..., description="Unique identifier for the user")
    
    @validator('updates')
    def validate_updates(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one memory update is required')
        if len(v) > 100:
            raise ValueError('Maximum batch size is 100 memory updates')
        return v

class MemoryQueryRequest(BaseModel):
    """
    Request model for querying memories.
    """
    user_id: str = Field(..., description="Unique identifier for the user")
    query: Optional[str] = Field(None, description="Search query for finding memories")
    tags: Optional[List[str]] = Field(None, description="Filter memories by tags")
    limit: Optional[int] = Field(10, description="Maximum number of memories to return")
    
    @validator('limit')
    def limit_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Limit must be a positive integer')
        return v

class MemoryStatsRequest(BaseModel):
    """
    Request model for getting memory statistics.
    """
    user_id: str = Field(..., description="Unique identifier for the user")

class ConfigUpdateRequest(BaseModel):
    """
    Request model for updating configuration.
    """
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    ollama_url: Optional[str] = None
    embedding_url: Optional[str] = None
    qdrant_host: Optional[str] = None
    qdrant_port: Optional[int] = None
    api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    ha_url: Optional[str] = None
    ha_token: Optional[str] = None
    memory_threshold: Optional[float] = None
    feedback_enabled: Optional[bool] = None
    learning_rate: Optional[float] = None
    features: Optional[Dict[str, bool]] = None
    force_recreate_collection: Optional[bool] = Field(False, description="Whether to recreate the vector collection if dimensions don't match")
    
    @validator('memory_threshold')
    def threshold_between_0_and_1(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Memory threshold must be between 0 and 1')
        return v
    
    @validator('learning_rate')
    def learning_rate_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Learning rate must be a positive number')
        return v
