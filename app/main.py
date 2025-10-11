# main.py
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Application imports
from app.config import *
from app.logging_config import (
    LoggingConfig,
    get_audit_logger,
    get_metrics_logger,
    logging_config,
    setup_logging,
)
from app.models.generator import *
from app.models.query_handler import *
from app.models.retriever import *
from app.run import *
from app.services.service_factory import ServiceFactory
from app.utils.preprocessing import *
from LoggingPipeline.Logging.logSystem import LogStatus

# Database imports
from pymilvus import connections, utility
from pymilvus.exceptions import MilvusException

# Initialize logging
logger = logging.getLogger(__name__)

# Global variables for resources
app_resources = None

class AppResources:
    def __init__(
        self, 
        retriever: object, 
        generator: Optional[object] = None, 
        user_chains: Optional[object] = None,
        metrics_logger: Optional[object] = None,
        audit_logger: Optional[object] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.user_chains = user_chains or {}
        self.metrics_logger = metrics_logger
        self.audit_logger = audit_logger

def get_resources() -> AppResources:
    return app.state.resources

# Input query structure
class QueryInput(BaseModel):
    query: str
    user_info: Dict

# Input sentence structure
class SentenceInput(BaseModel):
    text: str
    user_info: Dict

# initialize the resources
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize logging system
    metrics_logger, audit_logger = setup_logging(
        log_dir="logs",
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
    
    log.info("Initializing application resources...")
    start_time = time.time()
    
    try:
        # Load Retriever
        milvus_uri = MILVUS_URI
        milvus_collection = MILVUS_COLLECTION
        es_uri = ES_URI
        es_index = ES_INDEX
        
        # Initialize retriever with metrics logger
        rag_retriever = RAG_Retriever(
            milvus_uri, 
            milvus_collection, 
            es_uri, 
            es_index
        )
        retriever = rag_retriever.retriever
        
        # Log retriever initialization
        log.info("Retriever initialized successfully")
        
        # Load Generator
        rag_generator = RAG_Generator()
        generator = rag_generator.llm
        log.info("Generator initialized successfully")
        
        # Initialize user chains
        user_chains = {}
        
        # Initialize resources with loggers
        app.state.resources = AppResources(
            retriever=retriever,
            generator=generator,
            user_chains=user_chains,
            metrics_logger=metrics_logger,
            audit_logger=audit_logger
        )
        
        # Log successful initialization
        init_time = (time.time() - start_time) * 1000
        log.info(f"All resources initialized in {init_time:.2f}ms")
        
        if metrics_logger:
            metrics_logger.log_api_request(
                endpoint="/startup",
                platform="server",
                method="STARTUP",
                status_code=200,
                status=RequestStatus.SUCCESS,
                latency_ms=init_time,
                input_token_count=0,
                output_token_count=0
            )
            
    except Exception as e:
        log.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        if 'metrics_logger' in locals():
            metrics_logger.log_api_request(
                endpoint="/startup",
                platform="server",
                method="STARTUP",
                status_code=500,
                status=RequestStatus.FAILURE,
                latency_ms=(time.time() - start_time) * 1000,
                input_token_count=0,
                output_token_count=0,
                error_type=ErrorType.INTERNAL_ERROR
            )
        raise

    try:
        yield
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


def get_user_chain(user_info: Dict, resources: AppResources):
    user_id = user_info.get("username")
    print("User chains - ", resources.user_chains)
    if user_id not in resources.user_chains:
        print(f"Creating new chain for user {user_id}")
        # summary_generator = deepcopy(resources.generator)
        # summary_generator.max_tokens = 200
        # TODO: Try with ConversationBufferWindowMemory/ ConversationTokenBufferMemory/  ConversationSummaryMemory
        memory = ConversationSummaryMemory(
            llm=resources.generator,
            memory_key="chat_history",   # <-- memory injects into this
            return_messages=True,
            output_key="output",
            max_token_limit=500
        )
        print("\n\n######### Memory intialized!")
        qa_chain = setup_rag_pipeline_with_subqueries(user_info, resources.retriever, resources.generator, memory)
        
        user_chain_info = {}
        user_chain_info['chain'] = qa_chain
        user_chain_info['memory'] = memory
        log.info("User Info Saved")
        resources.user_chains[user_id] = user_chain_info

    return resources.user_chains[user_id]

def set_user_chain(user_info: Dict, qa_chain_info, resources: AppResources):
    user_id = user_info.get("username")
    resources.user_chains[user_id] = qa_chain_info


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging for health checks and metrics
    skip_paths = ["/healthz", "/metrics", "/favicon.ico"]
    if any(request.url.path.startswith(path) for path in skip_paths):
        return await call_next(request)
    
    # Generate request ID if not present in headers
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.time()
    
    # Get client information
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    referer = request.headers.get("referer", "")
    
    # Prepare request log data
    request_data = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params) if request.query_params else None,
        "client_ip": client_host,
        "user_agent": user_agent,
        "referer": referer or None,
        "content_type": request.headers.get("content-type"),
    }
    
    # Log request body (for non-GET requests) with size limit
    content_type = request.headers.get("content-type", "")
    if request.method != "GET" and "application/json" in content_type:
        try:
            body = await request.body()
            if body:
                # Limit request body size to 1KB for logging
                request_data["request_body"] = body.decode()[:1024]
        except Exception as e:
            log.warning(f"Failed to read request body: {str(e)}", 
                       request_id=request_id)
    
    log.info("Incoming request", **request_data)
    
    # Process the request
    response = None
    try:
        # Add request ID to request state for use in endpoints
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        # Prepare response log data
        response_data = {
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": round(process_time, 2),
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
        }
        
        # Log successful request
        log.info("Request completed", **response_data)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Log to metrics if available
        if hasattr(request.app.state, 'resources') and request.app.state.resources.metrics_logger:
            try:
                request.app.state.resources.metrics_logger.log_api_request(
                    endpoint=request.url.path,
                    platform=user_agent or "unknown",
                    method=request.method,
                    status_code=response.status_code,
                    status=RequestStatus.SUCCESS if response.status_code < 400 else RequestStatus.FAILURE,
                    latency_ms=process_time,
                    input_token_count=len(str(request_data.get("request_body", "")).split()),
                    output_token_count=0,  # Can be updated if response body is available
                    error_type=None if response.status_code < 400 else ErrorType.INTERNAL_ERROR,
                    request_id=request_id
                )
            except Exception as e:
                log.error(f"Failed to log metrics: {str(e)}", 
                         request_id=request_id, 
                         exc_info=True)
        
        return response
        
    except HTTPException as http_exc:
        # Handle HTTP exceptions (FastAPI's way of returning error responses)
        process_time = (time.time() - start_time) * 1000
        status_code = http_exc.status_code
        
        log_data = {
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(process_time, 2),
            "error": str(http_exc.detail) if hasattr(http_exc, 'detail') else str(http_exc),
            "error_type": "http_exception"
        }
        
        log.warning("Request completed with HTTP exception", **log_data)
        
        # Add request ID to response headers if response exists
        if response:
            response.headers["X-Request-ID"] = request_id
        
        # Re-raise the exception to let FastAPI handle it
        raise http_exc
        
    except Exception as exc:
        # Handle unexpected exceptions
        process_time = (time.time() - start_time) * 1000
        
        log_data = {
            "request_id": request_id,
            "status_code": 500,
            "duration_ms": round(process_time, 2),
            "error": str(exc),
            "error_type": "unhandled_exception"
        }
        
        log.error("Request failed with unhandled exception", 
                 **log_data, 
                 exc_info=True)
        
        # Log to metrics if available
        if hasattr(request.app.state, 'resources') and request.app.state.resources.metrics_logger:
            try:
                request.app.state.resources.metrics_logger.log_api_request(
                    endpoint=request.url.path,
                    platform=user_agent or "unknown",
                    method=request.method,
                    status_code=500,
                    status=RequestStatus.FAILURE,
                    latency_ms=process_time,
                    input_token_count=len(str(request_data.get("request_body", "")).split()),
                    output_token_count=0,
                    error_type=ErrorType.INTERNAL_ERROR,
                    request_id=request_id
                )
            except Exception as log_error:
                log.error(f"Failed to log error metrics: {str(log_error)}", 
                         request_id=request_id,
                         exc_info=True)
        
        # Return 500 error with request ID
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "request_id": request_id,
                "error": "Internal server error",
                "details": str(exc) if os.getenv("ENV", "production") != "production" else None
            },
            headers={"X-Request-ID": request_id}
        )

# Create a FastAPI instance
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: wide open; lock down later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HealthCheck
@app.get("/healthz")
async def health(resources: AppResources = Depends(get_resources)):
    """Health check endpoint that also logs the health check"""
    health_data = {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Log the health check
    if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
        try:
            resources.metrics_logger.log_api_request(
                endpoint="/healthz",
                platform="health_check",
                method="GET",
                status_code=200,
                status=RequestStatus.SUCCESS,
                latency_ms=0,  # Not measuring health check latency
                input_token_count=0,
                output_token_count=0
            )
        except Exception as e:
            log.error(f"Failed to log health check: {str(e)}", exc_info=True)
@app.get("/ingest")
async def ingest(resources: AppResources = Depends(get_resources)):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Log the start of ingestion
    log.info("Starting data ingestion", request_id=request_id)
    
    if hasattr(resources, 'audit_logger') and resources.audit_logger:
        try:
            resources.audit_logger.log_extraction(
                logtime=datetime.utcnow().isoformat(),
                ingestionTime=datetime.utcnow().isoformat(),
                sourceType="database",
                docId=request_id,
                status="started"
            )
        except Exception as e:
            log.error(f"Failed to log ingestion start: {str(e)}", exc_info=True)
    
    try:
        # Get database info from config
        database_info = {
            "database": "your_database",
            "username": "your_username",
            "password": "your_password",
            "url": "your_url",
            "port": 5432,
            "table": "your_table",
            "schema": "your_schema"
        }
        
        # Log before starting the ingestion
        log.info(f"Starting ingestion from database", request_id=request_id)
        
        # Perform the actual data ingestion
        ingest_data_with_db(
            database_info['database'], 
            database_info['username'], 
            database_info['password'], 
            database_info['url'], 
            database_info['port'], 
            database_info['table'], 
            database_info['schema']
        )
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log successful completion
        log.info("Data ingestion completed successfully", 
                request_id=request_id, 
                duration_ms=process_time)
        
        # Log to audit logger
        if hasattr(resources, 'audit_logger') and resources.audit_logger:
            try:
                resources.audit_logger.log_extraction(
                    logtime=datetime.utcnow().isoformat(),
                    ingestionTime=datetime.utcnow().isoformat(),
                    sourceType="database",
                    docId=request_id,
                    status="completed",
                    error=""
                )
            except Exception as e:
                log.error(f"Failed to log ingestion completion: {str(e)}", exc_info=True)
        
        # Log to metrics
        if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
            try:
                resources.metrics_logger.log_api_request(
                    endpoint="/ingest",
                    platform="ingestion",
                    method="GET",
                    status_code=200,
                    status=RequestStatus.SUCCESS,
                    latency_ms=process_time,
                    input_token_count=0,
                    output_token_count=0
                )
            except Exception as e:
                log.error(f"Failed to log metrics: {str(e)}", exc_info=True)
        
        return {
            "status": "success",
            "request_id": request_id,
            "duration_ms": process_time,
            "message": "Data ingestion completed successfully"
        }
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        error_msg = f"Ingestion failed: {str(e)}"
        log.error(error_msg, exc_info=True, request_id=request_id)
        
        # Log error to audit logger
        if hasattr(resources, 'audit_logger') and resources.audit_logger:
            try:
                resources.audit_logger.log_extraction(
                    logtime=datetime.utcnow().isoformat(),
                    ingestionTime=datetime.utcnow().isoformat(),
                    sourceType="database",
                    docId=request_id,
                    status="failed",
                    error=error_msg
                )
            except Exception as log_error:
                log.error(f"Failed to log ingestion error: {str(log_error)}", exc_info=True)
        
        # Log to metrics
        if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
            try:
                resources.metrics_logger.log_api_request(
                    endpoint="/ingest",
                    platform="ingestion",
                    method="GET",
                    status_code=500,
                    status=RequestStatus.FAILURE,
                    latency_ms=process_time,
                    input_token_count=0,
                    output_token_count=0,
                    error_type=ErrorType.INTERNAL_ERROR
                )
            except Exception as log_error:
                log.error(f"Failed to log error metrics: {str(log_error)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "request_id": request_id,
                "error": "Failed to ingest data",
                "details": str(e)
            }
        )
# Query Handler Endpoint 
@app.post("/query")
async def query(input: SentenceInput, resources: AppResources = Depends(get_resources)):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Log the start of query processing
    log.info("Starting query processing", request_id=request_id, user_info=input.user_info)
    
    if hasattr(resources, 'audit_logger') and resources.audit_logger:
        try:
            resources.audit_logger.log_query(
                logtime=datetime.utcnow().isoformat(),
                queryId=request_id,
                queryText=input.text,
                userId=input.user_info.get('user_id', 'unknown'),
                status="started"
            )
        except Exception as e:
            log.error(f"Failed to log query start: {str(e)}", exc_info=True)
    
    try:
        # Process the query
        qa_chain_info = get_user_chain(input.user_info, resources)
        memory = qa_chain_info['memory']
        response_dict, memory = check_sentence(input.text, memory=memory)
        set_user_chain(input.user_info, {'qa_chain': qa_chain_info['qa_chain'], 'memory': memory}, resources)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log successful completion
        log.info("Query processed successfully", 
                request_id=request_id, 
                user_info=input.user_info,
                duration_ms=process_time)
        
        # Log to audit logger
        if hasattr(resources, 'audit_logger') and resources.audit_logger:
            try:
                resources.audit_logger.log_query(
                    logtime=datetime.utcnow().isoformat(),
                    queryId=request_id,
                    queryText=input.text,
                    userId=input.user_info.get('user_id', 'unknown'),
                    status="completed",
                    response=response_dict,
                    error=""
                )
            except Exception as e:
                log.error(f"Failed to log query completion: {str(e)}", exc_info=True)
        
        # Log to metrics
        if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
            try:
                resources.metrics_logger.log_api_request(
                    endpoint="/query",
                    platform=input.user_info.get('platform', 'unknown'),
                    method="POST",
                    status_code=200,
                    status=RequestStatus.SUCCESS,
                    latency_ms=process_time,
                    input_token_count=len(input.text.split()),
                    output_token_count=len(str(response_dict).split())
                )
            except Exception as e:
                log.error(f"Failed to log metrics: {str(e)}", exc_info=True)
        
        # Add request ID to response
        if isinstance(response_dict, dict):
            response_dict["request_id"] = request_id
        
        return response_dict
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        error_msg = f"Query processing failed: {str(e)}"
        log.error(error_msg, exc_info=True, request_id=request_id, user_info=input.user_info)
        
        # Log error to audit logger
        if hasattr(resources, 'audit_logger') and resources.audit_logger:
            try:
                resources.audit_logger.log_query(
                    logtime=datetime.utcnow().isoformat(),
                    queryId=request_id,
                    queryText=input.text,
                    userId=input.user_info.get('user_id', 'unknown'),
                    status="failed",
                    error=error_msg
                )
            except Exception as log_error:
                log.error(f"Failed to log query error: {str(log_error)}", exc_info=True)
        
        # Log to metrics
        if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
            try:
                resources.metrics_logger.log_api_request(
                    endpoint="/query",
                    platform=input.user_info.get('platform', 'unknown'),
                    method="POST",
                    status_code=500,
                    status=RequestStatus.FAILURE,
                    latency_ms=process_time,
                    input_token_count=len(input.text.split()) if hasattr(input, 'text') else 0,
                    output_token_count=0,
                    error_type=ErrorType.INTERNAL_ERROR
                )
            except Exception as log_error:
                log.error(f"Failed to log error metrics: {str(log_error)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "request_id": request_id,
                "error": "Failed to process query",
                "details": str(e)
            }
        )
    
# Prediction endpoint
@app.post("/predict")
async def predict(input: QueryInput, resources: AppResources = Depends(get_resources)):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Log the start of prediction
    log.info("Starting prediction", request_id=request_id, user_info=input.user_info)
    
    if hasattr(resources, 'audit_logger') and resources.audit_logger:
        try:
            resources.audit_logger.log_prediction(
                logtime=datetime.utcnow().isoformat(),
                predictionId=request_id,
                queryText=input.query,
                userId=input.user_info.get('user_id', 'unknown'),
                status="started"
            )
        except Exception as e:
            log.error(f"Failed to log prediction start: {str(e)}", exc_info=True)
    
    try:
        # Process the prediction
        qa_chain_info = get_user_chain(input.user_info, resources)
        memory = qa_chain_info['memory']
        response_dict, memory = get_answer(
            input.query, 
            memory=memory, 
            retriever=resources.retriever, 
            llm=resources.generator
        )
        set_user_chain(input.user_info, {'qa_chain': qa_chain_info['qa_chain'], 'memory': memory}, resources)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log successful completion
        log.info("Prediction completed successfully", 
                request_id=request_id, 
                user_info=input.user_info,
                duration_ms=process_time)
        
        # Log to audit logger
        if hasattr(resources, 'audit_logger') and resources.audit_logger:
            try:
                resources.audit_logger.log_prediction(
                    logtime=datetime.utcnow().isoformat(),
                    predictionId=request_id,
                    queryText=input.query,
                    userId=input.user_info.get('user_id', 'unknown'),
                    status="completed",
                    response=response_dict,
                    error=""
                )
            except Exception as e:
                log.error(f"Failed to log prediction completion: {str(e)}", exc_info=True)
        
        # Log to metrics
        if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
            try:
                resources.metrics_logger.log_api_request(
                    endpoint="/predict",
                    platform=input.user_info.get('platform', 'unknown'),
                    method="POST",
                    status_code=200,
                    status=RequestStatus.SUCCESS,
                    latency_ms=process_time,
                    input_token_count=len(input.query.split()),
                    output_token_count=len(str(response_dict).split())
                )
            except Exception as e:
                log.error(f"Failed to log metrics: {str(e)}", exc_info=True)
        
        # Add request ID to response
        if isinstance(response_dict, dict):
            response_dict["request_id"] = request_id
        
        return response_dict
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        error_msg = f"Prediction failed: {str(e)}"
        log.error(error_msg, exc_info=True, request_id=request_id, user_info=input.user_info)
        
        # Log error to audit logger
        if hasattr(resources, 'audit_logger') and resources.audit_logger:
            try:
                resources.audit_logger.log_prediction(
                    logtime=datetime.utcnow().isoformat(),
                    predictionId=request_id,
                    queryText=input.query,
                    userId=input.user_info.get('user_id', 'unknown'),
                    status="failed",
                    error=error_msg
                )
            except Exception as log_error:
                log.error(f"Failed to log prediction error: {str(log_error)}", exc_info=True)
        
        # Log to metrics
        if hasattr(resources, 'metrics_logger') and resources.metrics_logger:
            try:
                resources.metrics_logger.log_api_request(
                    endpoint="/predict",
                    platform=input.user_info.get('platform', 'unknown'),
                    method="POST",
                    status_code=500,
                    status=RequestStatus.FAILURE,
                    latency_ms=process_time,
                    input_token_count=len(input.query.split()) if hasattr(input, 'query') else 0,
                    output_token_count=0,
                    error_type=ErrorType.INTERNAL_ERROR
                )
            except Exception as log_error:
                log.error(f"Failed to log error metrics: {str(log_error)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "request_id": request_id,
                "error": "Failed to process prediction",
                "details": str(e)
            }
        )
# # Define prediction endpoint
# @app.post("/run")
# def predict(input: QueryInput, res: AppResources = Depends(get_resources)):
#     log.info("Starting the Prediction...")
#     log.info("Input - " + str(input))
#     qa_chain = get_user_chain(input.user_info, res)
#     response_dict = run(input.text, qa_chain)
#     log.info("Response - " + str(response_dict))
#     response_dict["input"] = input.text
#     return response_dict
