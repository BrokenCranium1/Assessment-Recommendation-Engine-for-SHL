# SHL Assessment Recommendation System: Technical Approach

## 1. Problem Overview
Hiring managers and recruiters often face the "Cold Start" problem when selecting assessments for new roles. With SHL’s extensive catalog of **377+ individual test solutions**, finding the right balance between technical proficiency (Knowlege & Skills) and behavioral fit (Personality) is a manual, time-consuming process. 

Our objective was to build an intelligent, end-to-end recommendation system that:
- Accepts natural language queries or job description URLs.
- Returns **5-10 relevant assessments** in a standardized JSON format.
- Intelligently balances "Hard" (K-type) and "Soft" (P-type) skills.
- Achieves high **Mean Recall@10** on human-labeled ground truth data.

## 2. System Architecture

### 2.1 High-Level Design
The system implements a modern **Hybrid Retrieval-Augmented Generation (RAG)** framework. This architecture ensures that results are both contextually relevant (via embeddings) and precise (via keyword matching).

1.  **Data Layer**: A curated database of 377 assessments with descriptions, durations, and support features (Adaptive/Remote).
2.  **Retrieval Layer**: A two-stage process using **FAISS** for semantic similarity and **BM25** for exact match relevance.
3.  **Balance Layer**: An intent-classification layer that checks whether a query is about technical skills, behavioral traits, or both, and adjusts retrieval weights accordingly.
4.  **API Layer**: FastAPI backend exposing standardized endpoints that are easy to plug into a frontend.

### 2.2 Technology Stack
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Industry standard for AI/RAG applications. |
| **Web Scraping** | BeautifulSoup4, requests | Precise extraction of static and dynamic table content. |
| **Embeddings** | `all-MiniLM-L6-v2` | SOTA performance/speed ratio for local semantic search. |
| **Vector Search** | FAISS | Optimized for ultra-fast L2 distance similarity mapping. |
| **Keyword Search**| BM25 (rank_bm25) | Ensures "Hard Skill" terms (e.g., "Java", "SQL") are never missed. |
| **Backend** | FastAPI | Asynchronous, performance-first, with built-in Pydantic validation. |
| **Interface** | Vanilla HTML/CSS/JS | Focus on aesthetic excellence (Glassmorphism) without framework overhead. |

## 3. Data Pipeline

### 3.1 Scraping & Ingestion Strategy
The data pipeline was designed with a "Quality First" mantra, addressing three distinct challenges:

**1. Structural Table Detection**
Unlike many catalogs, SHL uses non-standard table containers. We bypassed fragile CSS selectors in favor of a **structural discovery algorithm**. The scraper identifies target tables by checking row-level headers for specific keywords ('Individual Test Solutions') and column configurations ('Pricing', 'Remote', 'Adaptive'). This allows the system to exclude "Pre-packaged Job Solutions" with 100% accuracy.

**2. Multi-Type Capture**
Assessments are rarely unidimensional. We implemented a loop to extract *all* badge identifiers (K, P, S, A, E) from each listing. This resulted in a dataset where **23.1% of assessments** are multi-typed, enabling more nuanced recommendations for complex queries.

**3. Deep Enrichment**
Baseline listings often lack details. We developed a secondary enrichment crawler that visits each assessment's detail page to extract terminal descriptions and duration metadata. This increased our semantic context significantly, allowing for better embedding clustering.

### 3.2 Dataset Integrity
- **Total Records**: 377 Individual Test Solutions.
- **Data Completeness**: 95%+ coverage for descriptions and durations.
- **Constraints**: 0% "Pre-packaged" solutions included, verified via substring audits.

## 4. Optimization & Performance Journey

The system's retrieval performance was optimized through four distinct iterations, measured against the **Mean Recall@10** metric.

### Stage 1: The Semantic Baseline (Mean Recall@10: 0.4462)
Initially, we used a pure dense retrieval approach. While it captured general intent (e.g., "soft skills"), it often missed specific technical tests like "Automata" because the embedding model generalized the name too far.

### Stage 2: Solving the URL Normalization Mismatch
In Stage 2, we identified a critical error: evaluation scores were artificially low because the training data used varying path formats (e.g., `/solutions/products/`) compared to our scraped terminal slugs (`/view/`).
- **The Fix**: Implemented a **"terminal-slug" normalization logic** across the engine and evaluation scripts.
- **Insight**: Found that 11 out of 54 training assessments are no longer in the live catalog, setting a realistic "Recall Ceiling" for this specific dataset.

### Stage 3: Hybrid Retrieval Breakthrough (Mean Recall@10: 0.5423)
By combining Semantic Search scores with BM25 keyword scores (50/50 split), Mean Recall@10 jumped by **21.5%**. This hybrid approach ensured that specific product names were prioritized while maintaining the ability to process natural language job descriptions.

### Stage 4: Optimized Tuning (Final Mean Recall@10: 0.5500)
Through grid-search tuning of weights, we discovered that a **0.7 Semantic / 0.3 Keyword** bias provided the best results. This allows the system to lean on semantic understanding for complex queries while retaining just enough keyword boosting to verify technical correctness.

## 5. Technical Challenges & Solutions

- **Challenge: Query Balancing**: Users often mix technical requirements (Java) with behavioral fit (Collaborative).
  - **Solution**: Implemented a "Boost" layer. If the engine detects a technical intent but retrieves only behavioral tests, it uses a similarity-threshold boost to ensure at least 2 results from each domain (K vs P) are included in the top 5.
- **Challenge: Dynamic Table Structure**: Paginated content with inconsistent start indexes.
  - **Solution**: Implemented a `start={}` parameter logic that iterates in steps of 12 until a "consecutive empty page" quality gate is hit, ensuring exhaustive catalog capture.

## 6. Conclusion & Results

The finalized SHL Assessment Recommendation System achieves a **Mean Recall@10 of 0.5500**, representing a **23.3% relative improvement** over the initial baseline. The system is reliable, handles structural URL variations cleanly, and provides a polished end-user experience through its responsive dashboard and standardized API. 

This journey demonstrates that for technical recommendation systems, a **Hybrid Search + Normalization** strategy is vital for bridging the gap between human user intent and raw catalog data.
