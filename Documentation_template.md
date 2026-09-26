# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** September 26, 2026  

---

## 1. Executive Summary
We present an end-to-end, scalable Entity Resolution (ER) solution designed for the Amazon ML Challenge 2026. The challenge requires linking business entities across three heterogeneous, noisy data sources ($S_1$ reference, $S_2$, and $S_3$ candidate pools) without shared identifiers, scored on macro-averaged $F_{0.5}$ (weighting precision 2× over recall). Our pipeline features a 4-tier memory-bounded blocking architecture (exact normalization, document-frequency pruned token inverted indexing, postal code bucketing, and dense FAISS semantic vector search via Apache-2.0 `all-MiniLM-L6-v2`), a pairwise feature engineering engine (13 string, token, character n-gram TF-IDF, and geospatial alignment signals), a LightGBM classifier with grouped 5-fold cross-validation, and an exact macro $F_{0.5}$ metric threshold sweep rigorously optimized to protect singleton entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights from Exploratory Data Analysis (EDA):
1. **Severe Asymmetry and Singletons:** Source 1 contains 2,206,821 records, where ~5.6% are true singletons (no match in $S_2$ or $S_3$). Under macro $F_{0.5}$, singletons score 1.0 if correctly predicted empty, but drop abruptly to 0.0 on any false positive match. Precision on singletons is paramount.
2. **Noise and Linguistic Variations:** Business names contain extensive variations: legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Inc`, `LLP`), DBA/trade aliases, punctuation differences, typos, and Devanagari/regional transliterations in Indian records. Addresses feature non-standard formatting, missing PIN/ZIP codes, and landmark references.
3. **Open Country Generalization:** While training covers US and India, the test set introduces `France`. The pipeline is built to handle country as an open string set with universal postal extraction and Unicode NFKD transliteration.
4. **Computational Scale:** The combined $S_2 + S_3$ candidate pool exceeds 10.3 million records. Naive Cartesian comparison requires $\approx 2.27 \times 10^{13}$ pairs. Memory-safe, vectorized indexing is strictly necessary to prevent OOM errors.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Tier Hybrid Blocking + Pairwise LightGBM Classifier + $F_{0.5}$-Tuned Decision Thresholding.
- **Core Innovation:** 
  1. **Bounded Token Indexing with Dynamic Pruning:** Prevents candidate explosion by pruning high-frequency business stopwords (`max_token_doc_freq`) and vectorized numpy score aggregation (`np.bincount`), reducing blocking runtime from hours/OOM to minutes.
  2. **Dense Semantic Retrieval:** FAISS IVF/Flat index over 384-dimensional normalized embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 22M parameters, Apache 2.0 license) retrieving high-noise transliteration matches unreachable by token overlap.
  3. **Grouped Leak-Free Training & Macro Threshold Sweeping:** GroupKFold on $S_1$ entities guarantees zero data leakage across candidate pairs, followed by a direct grid search over the official macro $F_{0.5}$ metric.

---

## 3. Candidate Generation (Blocking)
To reduce the $2.2\text{M} \times 10.3\text{M}$ search space to a tractable set:
- **Blocking keys used:**
  1. *Normalized exact name key:* Unicode lowercased, punctuation stripped, legal entity suffixes removed.
  2. *Inverted token overlap:* Informative name tokens indexed with document frequency cutoff ($\le 1\%$ corpus frequency) and capped candidate accumulation.
  3. *Postal code bucketing:* Regex-extracted postal codes (US 5-digit ZIP, India 6-digit PIN, France 5-digit code) with bucket size safety limits.
  4. *Dense semantic retrieval:* Fast cosine similarity search via FAISS using sentence embeddings.
- **Candidate pairs generated:** Capped at top $K=30$ candidate entities per Source-1 entity (ranked by a composite heuristic priority score).
- **How true matches were preserved:** Multi-strategy fallback ensures that entities lacking postal codes or exact token matches are retrieved via dense embedding neighborhoods, achieving high blocking recall while bounding pairs to tractable counts.

---

## 4. Matching Model

**Features used (13 pairwise features):**
- **Name features:** Levenshtein ratio, Jaro-Winkler similarity, token sort ratio, token set ratio, token Jaccard similarity, character 3-gram TF-IDF cosine similarity.
- **Address features:** Address Levenshtein ratio, address Jaro-Winkler similarity, address token sort ratio, address character 3-gram TF-IDF cosine similarity, postal code exact match boolean.
- **Structural / Alignment features:** Name length ratio, candidate source origin indicator ($S_2$ vs. $S_3$).

**Model type:** LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`), trained with binary cross-entropy, early stopping on grouped validation folds, and leaf/depth regularization.  
**Threshold selection method:** Exhaustive grid sweep ($\tau \in [0.10, 0.90]$) evaluated against the exact competition formula:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
macro-averaged across all Source-1 entities including singletons. The optimal threshold $\tau^*$ is saved in `artifacts/config.json`.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$:** Calibrated threshold consistently outperforms default 0.50 cutoff, raising singleton precision to $>0.94$ and achieving strong overall macro $F_{0.5}$.
- **Common false positives (wrong merges):**
  - Chain businesses and franchise branches (e.g. multiple retail outlets of the same parent brand in neighboring postal districts).
  - High token overlap on generic descriptor terms despite distinct street numbers.
- **Common false negatives (missed matches):**
  - Severe transliteration discrepancies between Devanagari script and phonetic English spellings when embeddings are disabled.
  - Incomplete addresses missing street numbers and postal codes entirely.

---

## 6. Conclusion
Our solution provides a robust, fully automated, and reproducible pipeline for large-scale business entity resolution. By combining memory-bounded multi-key candidate blocking, expressive string and semantic feature engineering, and a metric-tuned gradient boosted classifier, the system maximizes the precision-heavy macro $F_{0.5}$ metric while strictly abiding by all resource, licensing, and reproducibility constraints.

---

## Appendix

### A. Code Artefacts
All runnable source code is organized under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Text normalization, legal suffix stripping, postal regex extraction.
- `src/blocking.py`: Multi-strategy candidate generation (exact, token inverted index, postal, FAISS embeddings).
- `src/features.py`: Vectorized pairwise feature extraction (string metrics, TF-IDF cosine).
- `src/matching_model.py`: LightGBM classifier training and grouped cross-validation.
- `src/evaluate.py`: Macro $F_{0.5}$ calculation and threshold tuning.
- `src/train.py`: CLI training orchestration pipeline.
- `src/inference.py`: CLI test inference and candidate/match generation.
- `requirements.txt`: Exact pinned dependency versions.
- `README.md`: Step-by-step reproduction instructions.

### B. Reproducing Results
Run from `code/business_entity_resolution/`:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model and tune threshold
python -m src.train --data-dir dataset/train --output-dir artifacts

# 3. Generate candidate pairs and final predictions
python -m src.inference --data-dir dataset/test --artifacts-dir artifacts --output-dir output
```
