# main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from app.config import *
from app.run import run
from app.data.ingest import ingest_data
from pymilvus import connections
from pymilvus.exceptions import MilvusException
from pymilvus import connections, utility
import time
import os
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory
)
import runpod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from langchain_milvus import Milvus
import app.config as config
from app.models.retriever import *
from app.models.generator import *
from app.models.query_handler import *
from app.utils.preprocessing import *
from app.tests.test_values import *


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
log = logging.getLogger("api")


@dataclass
class AppResources:
    retriever: object
    generator: Optional[object] = None
    query_handler: Optional[object] = None

# Input query structure
class QueryInput(BaseModel):
    query: str
    user_info: Dict

# Input sentence structure
class SentenceInput(BaseModel):
    text: str

# Global chain cache
user_chains = {}
def get_user_chain(user_info: Dict, resources: AppResources):
    user_id = user_info.get("username")
    if user_id not in user_chains:
        print(f"Creating new chain for user {user_id}")
        # TODO: Try with ConversationBufferWindowMemory/ ConversationTokenBufferMemory/  ConversationSummaryMemory
        memory = ConversationSummaryMemory(
            llm=resources.generator,  # Need LLM for summarization
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        qa_chain = setup_rag_pipeline_with_subqueries(user_info, resources.retriever, resources.generator, memory)
        user_chains[user_id] = qa_chain
    return user_chains[user_id]


# Loading the resources
def load_resources():

    # Load Retriever
    milvus_uri = config.MILVUS_URI
    milvus_collection = config.MILVUS_COLLECTION
    es_uri = config.ES_URI
    es_index = config.ES_INDEX
    rag_retriever = RAG_Retriever(milvus_uri, milvus_collection, es_uri, es_index)
    # vectorstore = rag_retriever.vectorstore
    retriever = rag_retriever.retriever
    log.info("Retriever initialized")

    # Load Generator
    rag_generator = RAG_Generator()
    generator = rag_generator.llm
    log.info("Generator initiaized")

    # Put everything on app.state
    resources = AppResources(retriever=retriever, generator=generator)
    log.info("All resources initialized.")

    return resources

# Initialize the resources
resources = load_resources()

# HealthCheck
def health():
    return {"status": "ok"}

# Query Handler Endpoint 
def query(input: SentenceInput):
    log.info("Checking the sentence...")
    log.info("Input - " + str(input))
    response_dict = check_sentence(input.text)
    log.info("Response - " + str(response_dict))
    response_dict["input"] = input.text
    return response_dict
    
# Prediction endpoint
def predict(input: QueryInput):
    log.info("Starting the Prediction...")
    log.info("Input - " + str(input))
    qa_chain = get_user_chain(input.user_info, resources)
    response_dict = get_query_results(input.query, qa_chain)
    log.info("Response - " + str(response_dict))
    response_dict["input"] = input.text
    return response_dict

# Endpoint router
def route(event):
    log.info(f"Path - {path}")
    path = event.get("endpoint") or event.get("path")  

    if path == "/query":
        return query(event)
    if path == "/predict":
        return predict(event)
    elif path == "/healthz":
        return health()
    else:
        return {"error": "Unknown endpoint", "path": path}

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": route})