"""
Patterns API Endpoints

Bietet Zugriff auf erkannte Patterns und wiederkehrende Themen.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.memory.pattern_detector import (
    get_pattern_tracker,
    Pattern
)
from backend.core.component_cache import get_cached_components

logger = logging.getLogger("lexi_middleware.patterns_api")
router = APIRouter()


# Response Models
class PatternResponse(BaseModel):
    """Response model für Pattern"""
    id: str
    user_id: str
    pattern_type: str
    name: str
    description: str
    confidence: float
    frequency: int
    first_seen: str
    last_seen: str
    related_memory_ids: List[str] = []
    keywords: List[str] = []
    trend: str
    metadata: dict = {}


def pattern_to_response(pattern: Pattern) -> PatternResponse:
    """Konvertiert Pattern zu Response Model"""
    return PatternResponse(
        id=pattern.id,
        user_id=pattern.user_id,
        pattern_type=pattern.pattern_type,
        name=pattern.name,
        description=pattern.description,
        confidence=pattern.confidence,
        frequency=pattern.frequency,
        first_seen=pattern.first_seen.isoformat(),
        last_seen=pattern.last_seen.isoformat(),
        related_memory_ids=pattern.related_memory_ids,
        keywords=pattern.keywords,
        trend=pattern.trend,
        metadata=pattern.metadata
    )


@router.get("/patterns", response_model=List[PatternResponse])
async def get_patterns(
    user_id: str = Query(default="default", description="User ID"),
    pattern_type: Optional[str] = Query(None, description="Filter by type (topic, interest, behavior, routine)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    min_frequency: Optional[int] = Query(None, ge=1, description="Minimum frequency")
):
    """
    Holt alle erkannten Patterns eines Benutzers.

    Optional: Filter nach Typ, Confidence, Frequency.
    """
    try:
        bundle = get_cached_components()
        tracker = get_pattern_tracker(bundle.vectorstore)

        # Hole Patterns (optional mit Type-Filter)
        patterns = tracker.get_all_patterns(user_id, pattern_type=pattern_type)

        # Apply additional filters
        if min_confidence is not None:
            patterns = [p for p in patterns if p.confidence >= min_confidence]

        if min_frequency is not None:
            patterns = [p for p in patterns if p.frequency >= min_frequency]

        # Sort by frequency (most common first)
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        logger.info(f"Retrieved {len(patterns)} patterns for user {user_id}")

        return [pattern_to_response(p) for p in patterns]

    except Exception as e:
        logger.error(f"Error retrieving patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{pattern_id}", response_model=PatternResponse)
async def get_pattern(
    pattern_id: str,
    user_id: str = Query(default="default", description="User ID")
):
    """Holt ein einzelnes Pattern"""
    try:
        bundle = get_cached_components()
        tracker = get_pattern_tracker(bundle.vectorstore)

        # Get all patterns and find the one with matching ID
        patterns = tracker.get_all_patterns(user_id)
        pattern = next((p for p in patterns if p.id == pattern_id), None)

        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        return pattern_to_response(pattern)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/patterns/{pattern_id}", status_code=204)
async def delete_pattern(
    pattern_id: str,
    user_id: str = Query(default="default", description="User ID")
):
    """Löscht ein Pattern"""
    try:
        bundle = get_cached_components()
        tracker = get_pattern_tracker(bundle.vectorstore)

        # Verify pattern exists and belongs to user
        patterns = tracker.get_all_patterns(user_id)
        pattern = next((p for p in patterns if p.id == pattern_id), None)

        if not pattern:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        success = tracker.delete_pattern(pattern_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete pattern")

        logger.info(f"Deleted pattern {pattern_id}")

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting pattern {pattern_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/stats/summary")
async def get_patterns_summary(
    user_id: str = Query(default="default", description="User ID")
):
    """
    Gibt Zusammenfassung der Patterns zurück.

    Statistiken über häufigste Themen, Trends, etc.
    """
    try:
        bundle = get_cached_components()
        tracker = get_pattern_tracker(bundle.vectorstore)

        all_patterns = tracker.get_all_patterns(user_id)

        # Count by type
        type_counts = {}
        for pattern in all_patterns:
            type_counts[pattern.pattern_type] = type_counts.get(pattern.pattern_type, 0) + 1

        # Count by trend
        trend_counts = {}
        for pattern in all_patterns:
            trend_counts[pattern.trend] = trend_counts.get(pattern.trend, 0) + 1

        # Top patterns by frequency
        top_patterns = sorted(all_patterns, key=lambda p: p.frequency, reverse=True)[:10]

        # Top keywords across all patterns
        all_keywords = []
        for pattern in all_patterns:
            all_keywords.extend(pattern.keywords)

        from collections import Counter
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(20)]

        # Average confidence
        avg_confidence = sum(p.confidence for p in all_patterns) / len(all_patterns) if all_patterns else 0.0

        return {
            "success": True,
            "user_id": user_id,
            "total_patterns": len(all_patterns),
            "type_distribution": type_counts,
            "trend_distribution": trend_counts,
            "average_confidence": round(avg_confidence, 2),
            "top_patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "frequency": p.frequency,
                    "trend": p.trend,
                    "confidence": p.confidence
                }
                for p in top_patterns
            ],
            "top_keywords": top_keywords
        }

    except Exception as e:
        logger.error(f"Error getting patterns summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/trending/now")
async def get_trending_patterns(
    user_id: str = Query(default="default", description="User ID"),
    limit: int = Query(default=10, ge=1, le=50, description="Max number of results")
):
    """
    Gibt aktuell trendende Patterns zurück.

    Patterns mit 'increasing' Trend und hoher Confidence.
    """
    try:
        bundle = get_cached_components()
        tracker = get_pattern_tracker(bundle.vectorstore)

        all_patterns = tracker.get_all_patterns(user_id)

        # Filter: increasing trend
        trending = [p for p in all_patterns if p.trend == "increasing"]

        # Sort by combination of confidence and frequency
        trending.sort(key=lambda p: (p.confidence * p.frequency), reverse=True)

        # Limit results
        trending = trending[:limit]

        logger.info(f"Found {len(trending)} trending patterns for user {user_id}")

        return {
            "success": True,
            "user_id": user_id,
            "count": len(trending),
            "patterns": [pattern_to_response(p) for p in trending]
        }

    except Exception as e:
        logger.error(f"Error getting trending patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/force-detect")
async def force_pattern_detection(
    user_id: str = Query(default="default", description="User ID")
):
    """
    Erzwingt sofortige Pattern Detection.

    Nützlich für Testing oder nach vielen neuen Memories.
    """
    try:
        from backend.memory.pattern_detector import PatternAnalyzer
        from backend.core.component_cache import get_cached_components

        bundle = get_cached_components()
        tracker = get_pattern_tracker(bundle.vectorstore)
        vectorstore = bundle.vectorstore

        # Hole alle Memories
        all_memories = vectorstore.get_all_entries(with_vectors=True)

        if not all_memories:
            return {
                "success": False,
                "message": "No memories found for pattern detection"
            }

        # Erkenne Patterns
        topic_patterns = PatternAnalyzer.detect_topic_patterns(
            memories=all_memories,
            min_cluster_size=3,
            similarity_threshold=0.75
        )

        interest_patterns = PatternAnalyzer.detect_interest_patterns(
            memories=all_memories,
            min_frequency=3
        )

        all_patterns = topic_patterns + interest_patterns

        # Speichere neue Patterns
        saved = 0
        for pattern in all_patterns[:20]:  # Max 20
            if tracker.save_pattern(pattern):
                saved += 1

        logger.info(f"Force-detected {saved} patterns for user {user_id}")

        return {
            "success": True,
            "message": f"Pattern detection completed",
            "patterns_detected": len(all_patterns),
            "patterns_saved": saved,
            "topic_patterns": len(topic_patterns),
            "interest_patterns": len(interest_patterns)
        }

    except Exception as e:
        logger.error(f"Error in force pattern detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
