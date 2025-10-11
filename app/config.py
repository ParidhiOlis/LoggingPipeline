# --- 1. Configuration ---
import os
from dotenv import load_dotenv

load_dotenv()


# Documents Config
DOC_LOCATION = "/Users/suhaas/Desktop/olis/rag/database/documents/sample_documents_one/"

# Vector DB Config
MILVUS_URI = "http://milvus:19530"
MILVUS_LOCAL_URI = "http://localhost:19530"
MILVUS_COLLECTION = "docs_collection"
ES_URI = "http://elasticsearch:9200"
ES_LOCAL_URI = "http://localhost:9200"
ES_INDEX = "docs_bm25"


# LLM Config
VLLM_USE_LOCAL = False

LLM_MODEL_NAME = "ministral/Ministral-3b-instruct"  # vLLM model name.
LLM_LLAMA_CPP = "app/llm_models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Quantized model
LLM_TEMP = 0.0
LLM_TOP_P = 0.2
LLM_MAX_TOKENS = 50
LLM_MAX_MODEL_LEN = 8192
LLM_MAX_CONTEXT_LEN = 4096

HF_TOKEN = os.getenv("HF_TOKEN")

# OPENAI API Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLAMAINDEX API Config
LLAMAINDEX_API_KEY = os.getenv("LLAMAINDEX_API_KEY")

# Embedding Model Config
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"

# Parsing Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retreival Config
VECTOR_LOADING_BATCH_SIZE = 100
VECTOR_RETRIEVER_TOP_K = 3
BM25_RETRIEVER_TOP_K = 3
RETRIEVER_THRESHOLD = 0.8
RETRIEVER_TOP_K = 5
WEIGHT_VECTOR = 0.5
WEIGHT_BM25 = 0.5
MIN_DOCS_REQUIRED = 0

USE_RERANKER = False
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 2
RERANKER_SCORE_THRESHOLD = 0.5

# Query Handler Config
USE_QUERY_DECOMPOSITION = True
QUERY_DECOMPOSITION_N = 3
SUBQUERY_TOP_K = 2

# Responses
status = {
    "SUCCESS": "SUCCESS",
    "FAIL": "FAIL"
}

messages = {"NONE": "NONE"}     # Default Message Response

# Role-based access
USE_RBAC_PREFILTER = True
USE_RBAC_POSTFILTER = False
TEST_USER_INFO = {
    'username': 'suhaask',
    'current_roles': ['swe'], 
    'current_groups': ['product', '_ALL_'], 
}

# Memory Config
USE_MEMORY_RETRIEVAL = True