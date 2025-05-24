---
title: "Project Requirements Document: Agentic RAG - Internal vs. External Knowledge"
dateCreated: 2025-05-06
dateModified: 2025-05-09
---

# Project Requirements Document: Agentic RAG - Internal vs. External Knowledge

## Purpose

To enhance the existing agentic RAG system by integrating external sources (e.g., Perplexity) with internal knowledge (e.g., vector store-based personalized data) into a seamless, unified answer engine. This improvement aims to:
- Solve the "trust gap" by anchoring external suggestions in personalized context.
- Maximize clarity by merging personalized insights with up-to-date external knowledge dynamically.
- Close loop follow-ups using history-aware mechanisms.

---

## Functional Requirements

### 1. **Internal Knowledge Workflow (Personalized RAG)**

- **Integration with Existing Vector Store**:
	- Ensure robust similarity search between user queries and internal knowledge (e.g., journal embeddings).
	- Must support theme extraction and match highly relevant personal data segments to prompt inputs.
- **Agent Query Formulation**:
	- Convert open-ended user queries into orchestrated similarity searches tailored to match vector embeddings.
	- Enable "quick answer" prompts to detect gaps and formulate concise follow-up steps.

### 2. **External Knowledge Workflow (Perplexity API Integration)**

- **External Triggering Mechanism**:
	- Dynamically identify points where external knowledge could fill gaps or add novelty to personalized insights.
	- Generate short, summarized prompts from the user query to enhance external system performance.
- **Bounding Search Scope**:
	- Restrict external results to high-quality, traceable suggestions (e.g., academic papers, authoritative sources).
	- Enable "confidence scoring" for ranking external results relative to internal embeddings' context.

### 3. **Unified Answer Merging**

- **Adapter Layer for Merging Paths**:
	- Combine responses from personalized RAG and Perplexity API into a single concise, actionable answer.
	- Ensure output emphasizes clarity by grounding external recommendations in relevant internal themes.
- **Prioritized Relevance Matching**:
	- Internal knowledge anchors core recommendations.
	- External insights supplement with thought-provoking, non-redundant contributions.

### 4. **History-Awareness Enhancements**

- **Follow-Up Query Resolution**:
	- Maintain session context for iterative prompts (e.g., question chains).
	- Allow users to refine, re-trigger, or expand on queries seamlessly without losing prior context.
- **Cross-Referencing History**:
	- Use past queries to enrich ongoing search paths and improve user-tailored relevance dynamically.

---

## Technical Requirements

### 1. **APIs And Infrastructure**

- **Perplexity API**:
	- Integrate external search API for concise, grounded knowledge retrieval.
	- Support short-session calls, ensuring prompt responses without latency bottlenecks.

- **Existing RAG System**:
	- Leverage existing backend for query-to-vector matching.
	- Ensure embeddings pipeline handles both journal entries and wide-context personalization robustly.

### 2. **Orchestration Layer**

- **Parallelized Query Handling**:
	- Ensure internal RAG and Perplexity queries operate simultaneously.
	- Merge intermediary results before surfacing final outputs.

### 3. **Output Formatting**

- **Obsidian Integration**:
	- Automatically write prioritized, unified outputs back into user journals.
	- Ensure markdown structuring is legible, clear, and actionable:

Example:

**Internal Knowledge Theme:** Scaling ethical AI
**External Insight Added:** Paper on regulatory frameworks for open-source AI ethics.

- **Scalability Features**:
	- Design modular data pipelines for future integration (e.g., additional sources).

---

## Milestones

1. **Task 1:**
	 - Add Perplexity API integration.
	 - Test dual-path processing (internal/external calls in parallel).
2. **Task 2:**
	 - Finalize merge logic for concise answer generation.
	 - Implement history-aware enhancements to track query chains.
3. **Task 3:**
	 - Refine output-to-Obsidian pipeline.
	 - Perform usability testing for iterative improvement based on real-life scenarios.
4. **Task 4:**
	 - Optimize confidence scoring and feedback loops for personalized recommendations.
	 - Review and finalize pipeline modularity for scalability.

---

## Closing Notes

This update transforms the RAG system from a standalone personalized agent into a dynamic, personalized assistant that bridges gaps between what users know and the knowledge they need. By syncing external and personalized insights seamlessly, the system ensures clarity, trust, and actionable resonance for every query.
