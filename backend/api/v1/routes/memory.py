from fastapi import APIRouter, HTTPException, Depends, Request
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.models.memory_entry import MemoryEntry
from backend.core.bootstrap import initialize_components
from backend.utils.validators import InputValidator
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone
from functools import lru_cache
import logging

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@lru_cache
def get_memory_interface():
    """Dependency to get the QdrantMemoryInterface instance.

    Uses lru_cache to create a singleton instance that is initialized only once
    and then reused for subsequent requests, saving resources and improving performance.
    """
    try:
        _, vectorstore, _, _, _ = initialize_components()
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize memory interface: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize memory interface")


class MemoryAddRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000, description="Memory content")
    tags: Optional[List[str]] = Field(None, description="Optional tags for categorization")
    source: Optional[str] = Field(None, max_length=500, description="Optional source identifier")


class MemoryResponse(BaseModel):
    id: UUID
    content: str
    category: str
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    relevance: float
    timestamp: str


class MemoryQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class MemoryDeleteRequest(BaseModel):
    id: UUID = Field(..., description="ID of the memory entry to delete")


class MemoryDeleteResponse(BaseModel):
    status: str
    message: str
    id: UUID


class MemoryForgetRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="User identifier")
    query: str = Field(..., min_length=1, max_length=1000, description="Topic/query to forget")
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Minimum similarity score for deletion")


class MemoryForgetResponse(BaseModel):
    status: str
    message: str
    deleted_count: int
    deleted_ids: List[str]


@router.post("/memory/add", response_model=MemoryResponse)
@limiter.limit("10/minute")  # SECURITY: Rate limit memory writes
async def add_memory(
    memory_request: MemoryAddRequest,
    request: Request,
    memory_interface: QdrantMemoryInterface = Depends(get_memory_interface)
):
    """
    Add a new memory entry to the vector store.

    Uses user_id from middleware for multi-user memory isolation.

    SECURITY:
    - Rate limited to 10 requests/minute to prevent DoS
    - Validates all inputs to prevent injection attacks
    - Sanitizes content and tags
    """
    try:
        # Get user_id from middleware
        user_id = getattr(request.state, 'user_id', 'default')
        logger.debug(f"Adding memory for user_id: {user_id}")

        # Validate content
        validated_content = InputValidator.validate_content(
            memory_request.content,
            field_name="content",
            max_length=10000,
            allow_html=False
        )

        # Validate tags
        validated_tags = InputValidator.validate_tag_list(memory_request.tags, max_tags=10, max_tag_length=50)

        # Validate source
        validated_source = "api"
        if memory_request.source:
            validated_source = InputValidator.validate_content(
                memory_request.source,
                field_name="source",
                max_length=100,
                allow_html=False
            )

        entry_id = uuid4()
        timestamp = datetime.now(timezone.utc)

        entry = MemoryEntry(
            id=str(entry_id),  # Convert UUID to string for MemoryEntry
            content=validated_content,
            timestamp=timestamp,
            user_id=user_id,  # User isolation
            tags=validated_tags,
            source=validated_source,
            relevance=1.0
        )

        # Store the entry (user_id is already in the entry object)
        memory_interface.store_entry(entry)

        logger.info(f"Successfully added memory entry with ID: {entry_id}")

        return MemoryResponse(
            id=entry_id,
            content=memory_request.content,
            category=entry.category or "uncategorized",
            tags=memory_request.tags,
            source=memory_request.source,
            relevance=1.0,
            timestamp=timestamp.isoformat()
        )
        
    except ValueError as e:
        logger.error(f"Validation error when adding memory: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error adding memory entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to add memory entry")


@router.post("/memory/query", response_model=List[MemoryResponse])
@limiter.limit("100/minute")  # SECURITY: Rate limit memory reads
async def query_memory(
    memory_request: MemoryQueryRequest,
    request: Request,
    memory_interface: QdrantMemoryInterface = Depends(get_memory_interface)
):
    """
    Query memory entries using semantic search.

    Uses user_id from middleware to filter results by user.

    SECURITY:
    - Rate limited to 100 requests/minute
    - Validates query input
    - Limits result size
    """
    try:
        # Get user_id from middleware
        user_id = getattr(request.state, 'user_id', 'default')
        logger.debug(f"Querying memories for user_id: {user_id}")

        # Validate query
        validated_query = InputValidator.validate_content(
            memory_request.query,
            field_name="query",
            max_length=1000,
            allow_html=False
        )

        # Validate top_k
        validated_top_k = InputValidator.validate_limit(memory_request.top_k, default=5, max_limit=50)

        memory_entries = memory_interface.query_memories(
            query=validated_query,
            top_k=validated_top_k,
            user_id=user_id  # Filter by user_id
        )

        # Convert MemoryEntry objects to MemoryResponse objects
        results = []
        for entry in memory_entries:
            try:
                results.append(MemoryResponse(
                    id=UUID(entry.id),
                    content=entry.content,
                    category=entry.category or "uncategorized",
                    tags=entry.tags,
                    source=entry.source,
                    relevance=entry.relevance or 1.0,
                    timestamp=entry.timestamp.isoformat()
                ))
            except ValueError as e:
                logger.warning(f"Skipping invalid entry with ID {entry.id}: {e}")
                continue

        logger.info(f"Query '{memory_request.query}' returned {len(results)} results")
        return results
        
    except ValueError as e:
        logger.error(f"Validation error in query: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to query memories")


@router.delete("/memory/{memory_id}", response_model=MemoryDeleteResponse)
@limiter.limit("10/minute")  # SECURITY: Rate limit deletions
async def delete_memory_by_path(
    memory_id: UUID,
    request: Request,
    memory_interface: QdrantMemoryInterface = Depends(get_memory_interface)
):
    """
    Delete a memory entry by ID (path parameter version).

    SECURITY:
    - Rate limited to 10 requests/minute
    - Validates UUID format
    """
    # Validate UUID
    InputValidator.validate_uuid(str(memory_id), field_name="memory_id")
    return _delete_memory_logic(memory_id, memory_interface)


@router.post("/memory/delete", response_model=MemoryDeleteResponse)
async def delete_memory(
    request: MemoryDeleteRequest,
    memory_interface: QdrantMemoryInterface = Depends(get_memory_interface)
):
    """Delete a memory entry by ID (request body version)."""
    return _delete_memory_logic(request.id, memory_interface)


def _delete_memory_logic(
    memory_id: UUID,
    memory_interface: QdrantMemoryInterface
) -> MemoryDeleteResponse:
    """Shared logic for deleting memory entries."""
    try:
        # FIX: Check if entry exists before deletion
        try:
            points = memory_interface.client.retrieve(
                collection_name=memory_interface.collection,
                ids=[str(memory_id)]
            )
            if not points:
                raise KeyError(f"Memory {memory_id} not found")
        except Exception:
            raise KeyError(f"Memory {memory_id} not found")

        # Delete if exists
        memory_interface.delete_entry(memory_id)
        
        logger.info(f"Successfully deleted memory entry with ID: {memory_id}")
        
        return MemoryDeleteResponse(
            status="success",
            message="Memory entry deleted successfully",
            id=memory_id
        )
        
    except ValueError as e:
        logger.error(f"Invalid ID format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {str(e)}")
    except KeyError:
        logger.warning(f"Attempted to delete non-existent memory ID: {memory_id}")
        raise HTTPException(status_code=404, detail="Memory entry not found")
    except Exception as e:
        logger.error(f"Error deleting memory entry {memory_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete memory entry")


@router.get("/memory/stats")
async def get_memory_stats(
    memory_interface: QdrantMemoryInterface = Depends(get_memory_interface)
):
    """
    Get memory statistics including total count and category breakdown.

    Returns:
        - total: Total number of memory entries
        - categories: Breakdown by category
        - last_access: Timestamp of last access
    """
    try:
        from backend.memory.adapter import get_memory_stats

        stats = get_memory_stats()

        return {
            "total": stats.get("total", 0),
            "categories": stats.get("categories", {}),
            "last_access": stats.get("last_access", datetime.now(timezone.utc).isoformat())
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        # Return empty stats instead of error for UI
        return {
            "total": 0,
            "categories": {},
            "last_access": datetime.now(timezone.utc).isoformat()
        }


@router.get("/memory/health")
async def health_check():
    """Health check endpoint for the memory service."""
    try:
        # You could add a simple test query here to verify the interface is working
        return {"status": "healthy", "service": "memory"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.post("/memory/forget", response_model=MemoryForgetResponse)
@limiter.limit("5/minute")  # SECURITY: Strict rate limit for bulk deletions
async def forget_memories(
    request: MemoryForgetRequest,
    http_request: Request,
    memory_interface: QdrantMemoryInterface = Depends(get_memory_interface)
):
    """
    Delete memories matching a semantic query (forget command).

    This endpoint searches for memories semantically similar to the query
    and deletes those above the similarity threshold.

    SECURITY:
    - Rate limited to 5 requests/minute (strict limit for bulk deletions)
    - Validates all inputs
    - Limits deletion scope (max 50 memories per request)

    Args:
        user_id: User identifier
        query: Topic/query to forget
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.75)

    Returns:
        Status and list of deleted memory IDs

    Example:
        POST /v1/memory/forget
        {
            "user_id": "default",
            "query": "Python programming",
            "similarity_threshold": 0.75
        }
    """
    try:
        # Validate query
        validated_query = InputValidator.validate_content(
            request.query,
            field_name="query",
            max_length=1000,
            allow_html=False
        )

        # Validate user_id
        from backend.memory.adapter import validate_user_id
        validated_user_id = validate_user_id(request.user_id)

        # Validate similarity threshold
        if not 0.0 <= request.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        # Import delete function
        from backend.memory.adapter import delete_memories_by_content

        # Delete memories
        deleted_ids = delete_memories_by_content(
            query=validated_query,
            user_id=validated_user_id,
            similarity_threshold=request.similarity_threshold
        )

        logger.info(f"Forget command: deleted {len(deleted_ids)} memories for user {validated_user_id}")

        return MemoryForgetResponse(
            status="success",
            message=f"Successfully deleted {len(deleted_ids)} memories matching '{request.query}'",
            deleted_count=len(deleted_ids),
            deleted_ids=deleted_ids
        )

    except ValueError as e:
        logger.error(f"Validation error in forget command: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error in forget command: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete memories")