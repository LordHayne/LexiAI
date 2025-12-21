"""
Hybrid Search - Combines semantic (dense) and keyword (sparse) search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both search methods.

Usage:
    >>> from backend.qdrant.hybrid_search import fuse_search_results
    >>>
    >>> # Dense results: [(id1, 0.9), (id2, 0.8), ...]
    >>> # Sparse results: [(id3, 0.7), (id1, 0.6), ...]
    >>>
    >>> fused = fuse_search_results(dense_results, sparse_results, k=60)
    >>> # [(id1, score), (id3, score), (id2, score), ...]
"""

import logging
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

logger = logging.getLogger("lexi_middleware.hybrid_search")


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) - combines multiple ranked result lists.

    RRF formula: score(id) = sum_over_lists( weight / (k + rank(id)) )

    PERFORMANCE: Robust to score scale differences between search methods.

    Args:
        ranked_lists: List of ranked results [[(id, score), ...], [(id, score), ...]]
        k: RRF constant (typical: 60)
        weights: Optional weights for each list (default: equal weights)

    Returns:
        Fused ranked list: [(id, fused_score), ...]

    Example:
        >>> dense = [("id1", 0.9), ("id2", 0.8)]
        >>> sparse = [("id3", 0.7), ("id1", 0.6)]
        >>> fused = reciprocal_rank_fusion([dense, sparse], k=60)
    """
    if not ranked_lists:
        return []

    # Default: equal weights
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(f"Weights length ({len(weights)}) must match ranked_lists length ({len(ranked_lists)})")

    # Compute RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)

    for idx, ranked_list in enumerate(ranked_lists):
        weight = weights[idx]

        for rank, (doc_id, original_score) in enumerate(ranked_list):
            # RRF score contribution from this list
            rrf_contribution = weight / (k + rank + 1)  # rank starts at 0
            rrf_scores[doc_id] += rrf_contribution

    # Sort by fused score (descending)
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    logger.debug(f"RRF fused {len(ranked_lists)} lists into {len(fused_results)} unique results")

    return fused_results


def weighted_score_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    normalize: bool = True
) -> List[Tuple[str, float]]:
    """
    Weighted score fusion - combines results by weighted score averaging.

    Args:
        ranked_lists: List of ranked results
        weights: Optional weights for each list (default: equal)
        normalize: Whether to normalize scores before fusion

    Returns:
        Fused ranked list: [(id, fused_score), ...]
    """
    if not ranked_lists:
        return []

    # Default: equal weights
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("Weights length must match ranked_lists length")

    # Normalize scores if requested
    if normalize:
        normalized_lists = []
        for ranked_list in ranked_lists:
            if not ranked_list:
                normalized_lists.append([])
                continue

            max_score = max(score for _, score in ranked_list)
            min_score = min(score for _, score in ranked_list)

            if max_score == min_score:
                # All scores identical - normalize to 1.0
                normalized = [(doc_id, 1.0) for doc_id, _ in ranked_list]
            else:
                # Min-max normalization to [0, 1]
                normalized = [
                    (doc_id, (score - min_score) / (max_score - min_score))
                    for doc_id, score in ranked_list
                ]

            normalized_lists.append(normalized)
    else:
        normalized_lists = ranked_lists

    # Compute weighted scores
    weighted_scores: Dict[str, List[float]] = defaultdict(list)

    for idx, ranked_list in enumerate(normalized_lists):
        weight = weights[idx]

        for doc_id, score in ranked_list:
            weighted_scores[doc_id].append(weight * score)

    # Average scores for each document
    fused_scores = {
        doc_id: sum(scores) / len(scores)
        for doc_id, scores in weighted_scores.items()
    }

    # Sort by fused score (descending)
    fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    logger.debug(f"Weighted fusion: {len(ranked_lists)} lists -> {len(fused_results)} unique results")

    return fused_results


def fuse_search_results(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    strategy: str = "rrf",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60
) -> List[Tuple[str, float]]:
    """
    Fuse dense (semantic) and sparse (keyword) search results.

    HYBRID SEARCH: Combines the best of both worlds.

    Args:
        dense_results: Results from semantic search [(id, score), ...]
        sparse_results: Results from keyword search [(id, score), ...]
        strategy: Fusion strategy ("rrf" or "weighted")
        dense_weight: Weight for dense results (0.0-1.0)
        sparse_weight: Weight for sparse results (0.0-1.0)
        rrf_k: RRF constant (only for "rrf" strategy)

    Returns:
        Fused ranked list: [(id, fused_score), ...]

    Example:
        >>> # Semantic search finds: "neural networks", "AI"
        >>> dense = [("mem1", 0.9), ("mem2", 0.8)]
        >>>
        >>> # Keyword search finds exact: "machine learning"
        >>> sparse = [("mem3", 0.7), ("mem1", 0.6)]
        >>>
        >>> # Hybrid: mem1 ranks highest (found in both!)
        >>> fused = fuse_search_results(dense, sparse)
    """
    if not dense_results and not sparse_results:
        logger.warning("Both dense and sparse results are empty")
        return []

    # Handle case where only one result type exists
    if not dense_results:
        logger.debug("Only sparse results available - returning sparse only")
        return sparse_results

    if not sparse_results:
        logger.debug("Only dense results available - returning dense only")
        return dense_results

    # Fuse based on strategy
    if strategy == "rrf":
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=rrf_k,
            weights=[dense_weight, sparse_weight]
        )
    elif strategy == "weighted":
        fused = weighted_score_fusion(
            [dense_results, sparse_results],
            weights=[dense_weight, sparse_weight],
            normalize=True
        )
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}. Use 'rrf' or 'weighted'")

    logger.info(f"Hybrid search fused {len(dense_results)} dense + {len(sparse_results)} sparse = {len(fused)} results")

    return fused
