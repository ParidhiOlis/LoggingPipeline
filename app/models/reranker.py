from typing import List, Optional
import numpy as np

from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

# app/models/reranker.py
from typing import Any, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import ConfigDict
from app import config

class CrossEncoderReranker(BaseDocumentCompressor):
    # ---- Declare ALL fields up front (Pydantic v2) ----
    model_name: str = config.RERANKER_MODEL_NAME
    model: Optional[Any] = None                # will hold a sentence-transformers CrossEncoder
    top_k: int = 8
    score_threshold: Optional[float] = None
    batch_size: int = 32
    device: Optional[str] = None

    # allow non-pydantic types (CrossEncoder) as fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pydantic v2 hook: set up runtime objects here instead of __init__
    def model_post_init(self, __context) -> None:
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)

    # LangChain calls this
    def compress_documents(self, documents: List[Document], query: str, **kwargs) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, d.page_content or "") for d in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)  # np.ndarray-like
        scores = np.asarray(scores, dtype=np.float32)

        order = np.argsort(-scores)  # descending
        kept: list[tuple[int, float]] = []
        for i in order:
            if self.score_threshold is not None and scores[i] < self.score_threshold:
                continue
            kept.append((i, float(scores[i])))
            if len(kept) >= self.top_k:
                break

        reranked: List[Document] = []
        for idx, s in kept:
            d = documents[idx]
            d.metadata = {**(d.metadata or {}), "rerank_score": s}
            reranked.append(d)
        return reranked
