"""
Batch operations for memory management.

This module provides functions for efficient batch processing of memory operations,
including storing, updating, and deleting multiple memories in a single operation.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from backend.config.feature_flags import FeatureFlags
from backend.memory.memory_bootstrap import get_predictor

# Setup logging
logger = logging.getLogger("lexi_middleware.memory_batch")

# Default batch size for processing
DEFAULT_BATCH_SIZE = 50
# Default number of parallel tasks
DEFAULT_MAX_CONCURRENT = 5

async def batch_store_memories(
    memories: List[Dict[str, Any]],
    vectorstore,
    embeddings,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
) -> List[Tuple[str, str]]:
    logger.info(f"Starting batch storage of {len(memories)} memories")

    results = []
    errors = []
    semaphore = asyncio.Semaphore(max_concurrent)
    predictor = get_predictor()

    if not FeatureFlags.is_enabled("batch_operations"):
        logger.warning("Batch operations feature is disabled")
        for memory in memories:
            try:
                memory_id, timestamp = await store_single_memory(
                    memory, vectorstore, embeddings, semaphore, predictor
                )
                results.append((memory_id, timestamp))
            except Exception as e:
                logger.error(f"Error storing memory: {str(e)}")
                errors.append(str(e))

        logger.info(f"Completed batch storage with {len(results)} successes, {len(errors)} failures")
        return results

    batches = [memories[i:i + batch_size] for i in range(0, len(memories), batch_size)]

    for batch_num, batch in enumerate(batches):
        logger.debug(f"Processing batch {batch_num+1}/{len(batches)} with {len(batch)} memories")
        tasks = [
            asyncio.create_task(store_single_memory(mem, vectorstore, embeddings, semaphore, predictor))
            for mem in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {str(result)}")
                errors.append(str(result))
            else:
                results.append(result)

    logger.info(f"Completed batch storage with {len(results)} successes, {len(errors)} failures")
    return results

async def store_single_memory(
    memory: Dict[str, Any],
    vectorstore,
    embeddings,
    semaphore: asyncio.Semaphore,
    predictor
) -> Tuple[str, str]:
    async with semaphore:
        content = memory.get("content")
        user_id = memory.get("user_id")
        tags = memory.get("tags", [])

        if not content or not user_id:
            raise ValueError("Memory must contain content and user_id")

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        try:
            category = predictor.predict_category(content)
        except Exception as e:
            logger.error(f"Category prediction failed: {str(e)}")
            category = "unkategorisiert"

        try:
            vector = embeddings.embed_query(content)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

        memory_doc = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "category": category,
            "tags": tags,
            "timestamp": timestamp,
            "vector": vector
        }

        collection_name = f"user_{user_id}_memories"
        try:
            await vectorstore.add_documents(
                documents=[memory_doc],
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"Error storing memory in vector database: {str(e)}")
            raise

        logger.debug(f"Successfully stored memory {memory_id} for user {user_id}")
        return memory_id, timestamp

async def batch_delete_memories(
    memory_ids: List[str],
    user_id: str,
    vectorstore,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> int:
    logger.info(f"Starting batch deletion of {len(memory_ids)} memories for user {user_id}")

    if not FeatureFlags.is_enabled("batch_operations"):
        logger.warning("Batch operations feature is disabled")

    deleted_count = 0
    collection_name = f"user_{user_id}_memories"
    batches = [memory_ids[i:i + batch_size] for i in range(0, len(memory_ids), batch_size)]

    for batch_num, batch in enumerate(batches):
        logger.debug(f"Deleting batch {batch_num+1}/{len(batches)} with {len(batch)} memories")
        try:
            result = await vectorstore.delete(
                ids=batch,
                collection_name=collection_name
            )
            batch_deleted = sum(1 for success in result.get("success", []) if success)
            deleted_count += batch_deleted
            logger.debug(f"Deleted {batch_deleted} memories in batch {batch_num+1}")
        except Exception as e:
            logger.error(f"Error deleting batch {batch_num+1}: {str(e)}")

    logger.info(f"Completed batch deletion with {deleted_count} memories deleted")
    return deleted_count

async def batch_update_memories(
    updates: List[Dict[str, Any]],
    user_id: str,
    vectorstore,
    embeddings,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
) -> int:
    logger.info(f"Starting batch update of {len(updates)} memories for user {user_id}")

    if not FeatureFlags.is_enabled("batch_operations"):
        logger.warning("Batch operations feature is disabled")

    updated_count = 0
    collection_name = f"user_{user_id}_memories"
    semaphore = asyncio.Semaphore(max_concurrent)
    batches = [updates[i:i + batch_size] for i in range(0, len(updates), batch_size)]

    for batch_num, batch in enumerate(batches):
        logger.debug(f"Updating batch {batch_num+1}/{len(batches)} with {len(batch)} memories")
        tasks = [
            asyncio.create_task(
                update_single_memory(update, user_id, collection_name, vectorstore, embeddings, semaphore)
            )
            for update in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in batch_results:
            if isinstance(result, bool) and result:
                updated_count += 1
            elif isinstance(result, Exception):
                logger.error(f"Error in update processing: {str(result)}")

    logger.info(f"Completed batch update with {updated_count} memories updated")
    return updated_count

async def update_single_memory(
    update: Dict[str, Any],
    user_id: str,
    collection_name: str,
    vectorstore,
    embeddings,
    semaphore: asyncio.Semaphore
) -> bool:
    async with semaphore:
        memory_id = update.get("id")
        if not memory_id:
            logger.error("Memory update missing ID")
            return False

        try:
            current_memory = await vectorstore.get(
                ids=[memory_id],
                collection_name=collection_name
            )

            if not current_memory or not current_memory.get("documents"):
                logger.warning(f"Memory {memory_id} not found for user {user_id}")
                return False

            memory_data = current_memory["documents"][0]
            updated_memory = memory_data.copy()

            if "content" in update and update["content"]:
                updated_memory["content"] = update["content"]
                updated_memory["vector"] = embeddings.embed_query(update["content"])

            if "tags" in update and update["tags"] is not None:
                updated_memory["tags"] = update["tags"]

            if "category" in update and update["category"]:
                updated_memory["category"] = update["category"]

            updated_memory["updated_at"] = datetime.now().isoformat()

            await vectorstore.update(
                ids=[memory_id],
                documents=[updated_memory],
                collection_name=collection_name
            )

            logger.debug(f"Successfully updated memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {str(e)}")
            return False
