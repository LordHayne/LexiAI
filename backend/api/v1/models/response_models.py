"""
Response models for the Lexi API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class MemoryEntry(BaseModel):
    """
    Model for a memory entry.
    """
    id: str = Field(..., description="Unique identifier for the memory")
    content: str = Field(..., description="Content of the memory")
    tag: Optional[str] = Field(None, description="Primary tag for the memory")
    timestamp: str = Field(..., description="Timestamp when the memory was created")
    relevance: Optional[float] = Field(None, description="Relevance score of the memory")

class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    response: str = Field(..., description="The response from the AI")
    memory_used: bool = Field(False, description="Whether memory was used in generating the response")
    source: str = Field("llm", description="Source of the response (memory, llm, fallback)")
    memory_entries: Optional[List[MemoryEntry]] = Field(None, description="Memory entries used in the response")
    turn_id: Optional[str] = Field(None, description="Unique identifier for the conversation turn")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request in seconds")
    timestamp: Optional[str] = Field(None, description="Timestamp when the response was generated")
    config_warning: Optional[str] = Field(None, description="Configuration warning if any")

class MemoryStoreResponse(BaseModel):
    """
    Response model for the memory store endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    memory_id: str = Field(..., description="Unique identifier for the stored memory")
    stored_at: str = Field(..., description="Timestamp when the memory was stored")

class MemoryQueryResponse(BaseModel):
    """
    Response model for the memory query endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    memories: List[MemoryEntry] = Field(..., description="List of memory entries")
    total_count: int = Field(..., description="Total number of memories found")

class CategoryCount(BaseModel):
    """
    Model for category counts in memory statistics.
    """
    persönlich: int = Field(0, description="Count of personal memories")
    fakten: int = Field(0, description="Count of factual memories")
    präferenzen: int = Field(0, description="Count of preference memories") 
    andere: int = Field(0, description="Count of other memories")

class MemoryStats(BaseModel):
    """
    Model for memory statistics.
    """
    total_entries: int = Field(..., description="Total number of memory entries")
    categories: CategoryCount = Field(..., description="Memory entries by category")
    last_updated: str = Field(..., description="Timestamp of the last memory update")
    storage_usage: str = Field(..., description="Storage usage of the memories")
    most_common_tags: List[str] = Field(..., description="Most common tags")

class MemoryStatsResponse(BaseModel):
    """
    Response model for the memory stats endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    stats: MemoryStats = Field(..., description="Memory statistics")

class PerformanceStatsResponse(BaseModel):
    """
    Response model for the performance statistics endpoints.
    """
    success: bool = Field(True, description="Whether the request was successful")
    component: str = Field(..., description="Component the statistics are for (e.g., 'memory_cache', 'vector_search')")
    stats: Dict[str, Any] = Field(..., description="Performance statistics for the component")
    message: Optional[str] = Field(None, description="Additional information about the performance statistics")

class ConfigResponse(BaseModel):
    """
    Response model for the configuration endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    config: Dict[str, Any] = Field(..., description="Current configuration")

class ConfigUpdateResponse(BaseModel):
    """
    Response model for the configuration update endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    updated: List[str] = Field(..., description="List of updated configuration keys")
    current_config: Dict[str, Any] = Field(..., description="Current configuration after update")

class ComponentStatus(BaseModel):
    """
    Model for component status in health check.
    """
    status: str = Field(..., description="Status of the component (ok, warning, error)")
    latency_ms: Optional[int] = Field(None, description="Latency of the component in milliseconds")
    model: Optional[str] = Field(None, description="Model used by the component")
    version: Optional[str] = Field(None, description="Version of the component")
    message: Optional[str] = Field(None, description="Additional status message")
    actual: Optional[int] = Field(None, description="Actual dimensions for embeddings")
    configured: Optional[int] = Field(None, description="Configured dimensions in the vector database")
    mismatch: Optional[bool] = Field(None, description="Whether there's a dimension mismatch")

class HealthResponse(BaseModel):
    """
    Response model for the health endpoint.
    """
    status: str = Field(..., description="Overall status of the API (ok, warning, error)")
    version: str = Field(..., description="API version")
    uptime: str = Field(..., description="API uptime")
    components: Dict[str, ComponentStatus] = Field(..., description="Status of individual components")
    memory_stats: Optional[Dict[str, Any]] = Field(None, description="Memory statistics")

class ErrorResponse(BaseModel):
    """
    Response model for errors.
    """
    success: bool = Field(False, description="Whether the request was successful")
    error_code: str = Field(..., description="Error code for the occurred error")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional error details")

class ModelInfo(BaseModel):
    """
    Model for information about an available model.
    """
    name: str = Field(..., description="Name of the model")
    size: Optional[str] = Field(None, description="Size of the model (e.g., '7B', '13B')")
    family: Optional[str] = Field(None, description="Model family (e.g., 'llama', 'mistral')")
    quantization: Optional[str] = Field(None, description="Quantization level if applicable")
    modified_at: Optional[str] = Field(None, description="Last modified timestamp")
    digest: Optional[str] = Field(None, description="Model digest/hash")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional model details")

class ModelsResponse(BaseModel):
    """
    Response model for the models endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    models: List[ModelInfo] = Field(..., description="List of available models")
    default_llm_model: str = Field(..., description="Currently configured default LLM model")
    default_embedding_model: str = Field(..., description="Currently configured default embedding model")

class BatchStoreResult(BaseModel):
    """
    Result information for a single memory in a batch operation.
    """
    memory_id: str = Field(..., description="Unique identifier for the stored memory")
    stored_at: str = Field(..., description="Timestamp when the memory was stored")

class BatchMemoryStoreResponse(BaseModel):
    """
    Response model for the batch memory store endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    results: List[BatchStoreResult] = Field(..., description="Results for each memory stored")
    total_processed: int = Field(..., description="Total number of memories processed")
    failed_count: int = Field(0, description="Number of memories that failed to store")
    errors: Optional[List[str]] = Field(None, description="Errors encountered during batch processing")

class BatchMemoryDeleteResponse(BaseModel):
    """
    Response model for the batch memory delete endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    deleted_count: int = Field(..., description="Number of memories successfully deleted")
    total_requested: int = Field(..., description="Total number of memories requested for deletion")
    message: Optional[str] = Field(None, description="Additional information about the operation")

class BatchMemoryUpdateResponse(BaseModel):
    """
    Response model for the batch memory update endpoint.
    """
    success: bool = Field(True, description="Whether the request was successful")
    updated_count: int = Field(..., description="Number of memories successfully updated")
    total_requested: int = Field(..., description="Total number of memories requested for update")
    failed_ids: Optional[List[str]] = Field(None, description="IDs of memories that failed to update")
    message: Optional[str] = Field(None, description="Additional information about the operation")
