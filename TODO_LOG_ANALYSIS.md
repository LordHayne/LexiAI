# Log Analysis TODOs

This list captures the remaining items from the log analysis, plus context for the next session.

## Open items (highest impact first)

1) Self-reflection gating (reduce LLM calls)
- Only run self-reflection on risk cases: factual claims, external actions, long/uncertain answers.
- Keep fast path for smart home and tool-free answers.
- Consider a small heuristic (length, presence of numbers/entities, web/tool usage).

2) Cluster rebuild persistence
- Avoid rebuilding clusters on every request.
- Persist cluster model and rebuild only when enough new memories added (e.g., +50) or on a nightly schedule.
- Look for the "Clusters not available - rebuilding automatically" path and add a cache + threshold.

3) Qdrant scroll reductions (remaining)
- We already limited chat history, memory stats, synthesizer, optimizer.
- Still review other scroll-heavy paths (e.g., memory_synthesizer legacy, pattern_detector, goal_tracker, knowledge_gap_detector, optimizer subflows).
- Prefer time filters (timestamp_ms) or recent-only windows with safe fallback.

4) UI auth enforcement (security)
- Option: set LEXI_UI_AUTH_REQUIRED=true and ensure UI sends credentials (API key/JWT or proxy auth).
- Alternative: enforce Basic Auth via reverse proxy.

## Context / defaults added
- persistent_config.json now includes: chat_history_days, stats_days, memory_synthesis_days, optimizer_days,
  fact_confidence, fact_min_confidence, fact_ttl_days, memory_fallback_threshold.
- workers_config.yaml includes recent_days for deduplication, relevance_reranking, data_quality.
- LEXI_CHAT_HISTORY_DAYS, LEXI_STATS_DAYS, LEXI_MEMORY_SYNTHESIS_DAYS, LEXI_OPTIMIZER_DAYS are now mapped in persistence.

## Notes for next session
- Check logs after changes to confirm:
  - fewer full scrolls
  - fewer self-reflection calls
  - cluster rebuilds not running per request
- If needed, add timestamp_ms backfill for legacy entries (optional migration script).
