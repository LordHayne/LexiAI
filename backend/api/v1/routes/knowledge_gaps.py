"""
Knowledge Gaps API Endpoints - Minimale Version
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.memory.knowledge_gap_detector import get_knowledge_gap_tracker, KnowledgeGap
from backend.core.component_cache import get_cached_components

logger = logging.getLogger("lexi_middleware.knowledge_gaps_api")
router = APIRouter()


class KnowledgeGapResponse(BaseModel):
    id: str
    gap_type: str
    title: str
    description: str
    suggestion: str
    priority: float
    confidence: float
    created_at: str
    dismissed: bool


@router.get("/knowledge-gaps", response_model=List[KnowledgeGapResponse])
async def get_knowledge_gaps(user_id: str = Query(default="default")):
    """Holt alle aktiven Wissenslücken"""
    try:
        bundle = get_cached_components()
        tracker = get_knowledge_gap_tracker(bundle.vectorstore)
        gaps = tracker.get_all_gaps(user_id, include_dismissed=False)
        
        # Sort by priority
        gaps.sort(key=lambda g: g.priority, reverse=True)
        
        return [KnowledgeGapResponse(
            id=g.id,
            gap_type=g.gap_type,
            title=g.title,
            description=g.description,
            suggestion=g.suggestion,
            priority=g.priority,
            confidence=g.confidence,
            created_at=g.created_at.isoformat(),
            dismissed=g.dismissed
        ) for g in gaps]
    except Exception as e:
        logger.error(f"Error retrieving knowledge gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-gaps/{gap_id}/dismiss", status_code=204)
async def dismiss_knowledge_gap(gap_id: str):
    """Markiert eine Wissenslücke als dismissed"""
    try:
        bundle = get_cached_components()
        tracker = get_knowledge_gap_tracker(bundle.vectorstore)
        
        success = tracker.dismiss_gap(gap_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Gap {gap_id} not found")
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dismissing gap {gap_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-gaps/stats/summary")
async def get_knowledge_gaps_summary(user_id: str = Query(default="default")):
    """Statistiken über Wissenslücken"""
    try:
        bundle = get_cached_components()
        tracker = get_knowledge_gap_tracker(bundle.vectorstore)
        
        all_gaps = tracker.get_all_gaps(user_id, include_dismissed=True)
        active_gaps = [g for g in all_gaps if not g.dismissed]
        
        type_counts = {}
        for gap in active_gaps:
            type_counts[gap.gap_type] = type_counts.get(gap.gap_type, 0) + 1
        
        avg_priority = sum(g.priority for g in active_gaps) / len(active_gaps) if active_gaps else 0.0
        
        return {
            "success": True,
            "total_gaps": len(all_gaps),
            "active_gaps": len(active_gaps),
            "dismissed_gaps": len(all_gaps) - len(active_gaps),
            "type_distribution": type_counts,
            "average_priority": round(avg_priority, 2)
        }
    except Exception as e:
        logger.error(f"Error getting gaps summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
