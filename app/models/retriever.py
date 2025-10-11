
import os
import time
from typing import List
import logging

from elasticsearch import Elasticsearch
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain.retrievers import EnsembleRetriever
from app.models.reranker import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from app.config import *
import os, torch
logger = logging.getLogger(__name__)


os.environ.pop("PYTORCH_DEFAULT_DEVICE", None)
try:
    torch.set_default_device("cpu")   # requires torch>=2.1; safe no-op if already CPU
except Exception:
    pass

if torch.cuda.is_available():
    device_type = "cuda" # Use the default CUDA device (cuda:0)
    # Or specify a specific GPU: device = torch.device("cuda:1")
else:
    device_type = "cpu"

class RAG_Retriever():

    def __init__(self, milvus_uri, milvus_collection, es_uri, es_index):
        
        self.embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={'device': device_type}
            )
        self.vectorstore = Milvus(
            embedding_function=self.embed_model,
            connection_args={"uri": milvus_uri},
            collection_name=milvus_collection,
            )
        print("&&&&&&& milvus vectorstore - ", self.vectorstore)
        vec_retriever = self.vectorstore.as_retriever(search_kwargs={"k": VECTOR_RETRIEVER_TOP_K, "score_threshold": RETRIEVER_THRESHOLD})
        es_client = Elasticsearch([es_uri])
        bm25_retriever = ElasticSearchBM25Retriever(
            client=es_client,
            index_name=es_index,
            k=BM25_RETRIEVER_TOP_K
            )
        hybrid_retriever = EnsembleRetriever(
            retrievers=[vec_retriever, bm25_retriever],
            weights=[WEIGHT_VECTOR, WEIGHT_BM25],
            )
        if USE_RERANKER is False:
            self.retriever = hybrid_retriever
        else:
            compressor = CrossEncoderReranker(
                model_name=RERANKER_MODEL_NAME,
                top_k=RERANK_TOP_K,
                score_threshold=RERANKER_SCORE_THRESHOLD,
            )
            self.retriever = ContextualCompressionRetriever(
                base_retriever=hybrid_retriever,
                base_compressor=compressor,
            )
