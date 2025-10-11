import os
import time
from typing import List
import logging

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone

import pinecone
from pinecone import Pinecone, ServerlessSpec
from app.config import *
from app.models.generator import load_llm
from app.models.retriever import *
from app.utils.preprocessing import *

logger = logging.getLogger(__name__)

# --- RAG Pipeline Setup ---
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] {d.page_content}\nSOURCE_ID: {d.metadata.get('doc_id', d.metadata.get('source','unknown'))}"
        for i, d in enumerate(docs)
    )

def has_enough_context(x):
    docs = x["docs"]
    return isinstance(docs, list) and len(docs) >= MIN_DOCS_REQUIRED

def setup_rag_pipeline(vectorstore: Pinecone) -> RetrievalQA:
    """Sets up the RetrievalQA chain."""
    start_time = time.time()
    logger.info("Setting up RAG pipeline")

    try:
        llm = load_llm()
        RAG_PROMPT = PromptTemplate.from_template("""You are a strict, grounded QA assistant. Use ONLY the information in <documents>.
If the answer is not present or not clearly supported, say exactly:
"I don’t know based on the provided documents."

<documents>
{context}
</documents>

Question: {question}
Answer:
""")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # or "refine", "map_reduce", "map_rerank"
            # retriever = vectorstore.as_retriever(search_kwargs={
            #     "k":RETRIEVER_TOP_K,
            #     "score_threshold": RETRIEVER_THRESHOLD,})
            retriever=vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": RETRIEVER_TOP_K,
                    "score_threshold": RETRIEVER_THRESHOLD,
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_PROMPT},
        )
    except Exception as e:
        logger.error(f"Error setting up RAG pipeline: {e}", exc_info=True)
        return None

    end_time = time.time()
    logger.info(f"RAG pipeline setup in {end_time - start_time:.2f} seconds")
    return qa_chain

# --- Querying ---
def query_rag_pipeline(qa_chain: RetrievalQA, query: str) -> dict:
    """Queries the RAG pipeline."""
    start_time = time.time()
    logger.info(f"Querying RAG pipeline with query: {query}")

    try:
        result = qa_chain({"query": query})
    except Exception as e:
        logger.error(f"Error querying the RAG pipeline: {e}", exc_info=True)
        return {}

    end_time = time.time()
    logger.info(f"Query completed in {end_time - start_time:.2f} seconds")
    return result






