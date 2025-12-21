"""
Sparse Vector Encoder for Hybrid Search.

Generates sparse vectors (TF-IDF based) for keyword-based search.
Combines with dense semantic vectors for hybrid search capabilities.

Usage:
    >>> from backend.embeddings.sparse_encoder import SparseVectorEncoder
    >>>
    >>> encoder = SparseVectorEncoder()
    >>> encoder.fit(["document 1", "document 2", ...])
    >>>
    >>> sparse_vector = encoder.encode("search query")
    >>> print(sparse_vector)
    >>> # {"indices": [12, 45, 78], "values": [0.8, 0.6, 0.4]}
"""

import re
import math
import logging
from typing import List, Dict, Set, Optional
from collections import Counter, defaultdict
from threading import RLock

logger = logging.getLogger("lexi_middleware.sparse_encoder")


class SparseVectorEncoder:
    """
    TF-IDF based sparse vector encoder for hybrid search.

    Generates sparse vectors where:
    - indices: Vocabulary term IDs
    - values: TF-IDF scores

    PERFORMANCE: Only non-zero values stored = efficient for Qdrant sparse vectors.
    """

    def __init__(self, max_features: int = 10000, min_df: int = 2, max_df: float = 0.8):
        """
        Initialize sparse vector encoder.

        Args:
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency (ignore rare terms)
            max_df: Maximum document frequency ratio (ignore common terms like "the")
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        # Vocabulary: term -> index
        self.vocabulary: Dict[str, int] = {}
        self.inverse_vocabulary: Dict[int, str] = {}

        # IDF scores: term -> score
        self.idf_scores: Dict[str, float] = {}

        # Document frequency: term -> count
        self.doc_freq: Counter = Counter()
        self.num_documents: int = 0

        # Thread safety
        self._lock = RLock()

        # Fitted flag
        self.is_fitted: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Input text

        Returns:
            List of tokens (lowercase, alphanumeric)
        """
        # Lowercase
        text = text.lower()

        # Extract alphanumeric tokens (2+ chars)
        tokens = re.findall(r'\b[a-z0-9]{2,}\b', text)

        return tokens

    def fit(self, documents: List[str]) -> 'SparseVectorEncoder':
        """
        Fit encoder on corpus of documents.

        Builds vocabulary and computes IDF scores.

        Args:
            documents: List of text documents

        Returns:
            Self (for chaining)

        Example:
            >>> encoder.fit(["doc 1", "doc 2", "doc 3"])
        """
        with self._lock:
            logger.info(f"Fitting sparse encoder on {len(documents)} documents...")

            # Reset
            self.doc_freq.clear()
            self.num_documents = len(documents)

            if self.num_documents == 0:
                logger.warning("No documents to fit - encoder remains unfitted")
                return self

            # Count document frequencies
            for doc in documents:
                tokens = self._tokenize(doc)
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self.doc_freq[token] += 1

            # Filter by document frequency
            filtered_terms = [
                term for term, freq in self.doc_freq.items()
                if freq >= self.min_df and freq <= self.max_df * self.num_documents
            ]

            # Limit vocabulary size
            if len(filtered_terms) > self.max_features:
                # Keep most frequent terms
                filtered_terms = sorted(filtered_terms, key=lambda t: self.doc_freq[t], reverse=True)[:self.max_features]

            # Build vocabulary
            self.vocabulary = {term: idx for idx, term in enumerate(sorted(filtered_terms))}
            self.inverse_vocabulary = {idx: term for term, idx in self.vocabulary.items()}

            # Compute IDF scores
            self.idf_scores = {}
            for term in self.vocabulary:
                df = self.doc_freq[term]
                # IDF = log(N / df)
                idf = math.log((self.num_documents + 1) / (df + 1)) + 1  # +1 for smoothing
                self.idf_scores[term] = idf

            self.is_fitted = True
            logger.info(f"✅ Sparse encoder fitted: {len(self.vocabulary)} terms in vocabulary")

            return self

    def encode(self, text: str, top_k: Optional[int] = 50) -> Dict[str, List]:
        """
        Encode text into sparse vector.

        Args:
            text: Input text to encode
            top_k: Keep only top K highest-scoring terms (None = all terms)

        Returns:
            Sparse vector dict: {"indices": [...], "values": [...]}

        Example:
            >>> vector = encoder.encode("machine learning frameworks")
            >>> print(vector)
            >>> # {"indices": [12, 45, 78], "values": [0.82, 0.61, 0.43]}
        """
        if not self.is_fitted:
            logger.warning("Encoder not fitted - returning empty sparse vector")
            return {"indices": [], "values": []}

        with self._lock:
            # Tokenize
            tokens = self._tokenize(text)

            # Count term frequencies
            tf = Counter(tokens)

            # Compute TF-IDF scores
            tf_idf_scores = {}
            for term, count in tf.items():
                if term in self.vocabulary:
                    # TF = count / total_tokens (normalized)
                    tf_score = count / len(tokens) if len(tokens) > 0 else 0

                    # TF-IDF = TF * IDF
                    idf_score = self.idf_scores.get(term, 0)
                    tf_idf = tf_score * idf_score

                    term_idx = self.vocabulary[term]
                    tf_idf_scores[term_idx] = tf_idf

            # Sort by score (descending)
            sorted_items = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep top K
            if top_k is not None and len(sorted_items) > top_k:
                sorted_items = sorted_items[:top_k]

            # Build sparse vector
            indices = [idx for idx, _ in sorted_items]
            values = [score for _, score in sorted_items]

            return {
                "indices": indices,
                "values": values
            }

    def get_stats(self) -> Dict[str, any]:
        """
        Get encoder statistics.

        Returns:
            Statistics dict
        """
        with self._lock:
            return {
                "is_fitted": self.is_fitted,
                "vocabulary_size": len(self.vocabulary),
                "num_documents": self.num_documents,
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df
            }


# Global encoder instance
_sparse_encoder: Optional[SparseVectorEncoder] = None
_encoder_lock = RLock()


def get_sparse_encoder(force_new: bool = False) -> SparseVectorEncoder:
    """
    Get global sparse encoder instance (singleton).

    Args:
        force_new: Create new instance instead of returning cached

    Returns:
        SparseVectorEncoder instance
    """
    global _sparse_encoder

    with _encoder_lock:
        if _sparse_encoder is None or force_new:
            _sparse_encoder = SparseVectorEncoder(
                max_features=10000,  # 10k terms vocabulary
                min_df=2,            # Ignore terms in <2 docs
                max_df=0.8           # Ignore terms in >80% docs
            )
            logger.info("Initialized global sparse encoder")

        return _sparse_encoder


def fit_sparse_encoder_on_corpus(documents: List[str]) -> SparseVectorEncoder:
    """
    Fit global sparse encoder on document corpus.

    Args:
        documents: List of text documents

    Returns:
        Fitted encoder

    Example:
        >>> from backend.embeddings.sparse_encoder import fit_sparse_encoder_on_corpus
        >>> encoder = fit_sparse_encoder_on_corpus(all_memory_contents)
    """
    encoder = get_sparse_encoder()
    encoder.fit(documents)
    return encoder
