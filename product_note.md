# Fermi Podcast Companion: Product Note

## The Problem
Learning from long-form audio is inherently difficult. Podcasts are dense with information, but they are linear, unsearchable, and difficult to reference. When a listener remembers an insight from an episode they heard weeks ago—or wants to compare how two different scientists approached a similar problem—finding that exact moment in hours of audio is nearly impossible.

The Fermi Podcast Companion solves this by transforming the *Great Papers* podcast series into an interactive, grounded learning assistant. It allows users to ask complex, cross-episode questions and immediately jump to the exact audio timestamp where the answer is discussed.

## Scope and Architecture Decisions
We scoped this product to a focused corpus of **16 core episodes**. This specific scale informed several intentional, evidence-driven architectural decisions:

1. **Fixed-Size Chunking:** We used a simple, deterministic time-based chunking strategy (6-minute chunks with a 1-minute overlap). While more complex strategies exist, the evaluation suite proved that standard chunking perfectly captured the necessary context for this corpus size.
2. **Episode Summaries for Recommendations:** Instead of relying entirely on granular chunks for broad recommendation queries (e.g., "Which episode covers biology?"), we introduced a parallel tier of offline-generated *Episode Summaries*. This allows the system to accurately recommend full episodes for high-level topics without confusing the factual chunk-retrieval pipeline.
3. **No Reranking or Complex Hierarchies:** We deliberately omitted advanced RAG techniques like cross-encoder reranking, parent-child document retrieval, or semantic chunking. 

## What We Left Out (And Why)
Engineering should be driven by evidence, not hype. At our 16-episode scale, the evaluation suite (21 rigorous test cases covering factual grounding, cross-episode comparisons, multi-turn tracking, and adversarial refusals) passed with a **100% success rate** using only standard cosine similarity and strict, deterministic prompt engineering.

Implementing rerankers or complex hierarchical retrieval would have added significant latency, infrastructure complexity, and compute costs without moving the needle on actual answer quality. If this companion were scaled to a meaningfully larger corpus (e.g., 500+ episodes across multiple podcasts), the signal-to-noise ratio in dense retrieval would drop, and those advanced techniques would be reconsidered. But for this product, we optimized for speed, simplicity, and bulletproof grounding over unnecessary complexity.

However, manual testing did expose a late gap handling compound/multi-topic queries (e.g., comparing two completely unrelated concepts), which required adding a lightweight LLM-based query decomposition step. If given more time or a larger corpus, more complex multi-hop query decomposition would be the next architectural piece to strengthen.
