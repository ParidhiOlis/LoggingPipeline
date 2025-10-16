import os
import time
from typing import List
import logging
import logging
from app import config
from app.models.parser import *
from sqlalchemy import create_engine, MetaData, Table, text
from datetime import datetime
from typing import List, Any


logger = logging.getLogger(__name__)

class VectorDBHandler():
    def __init__(self, host, port):
        pass
    
    def conenct_to_vectorstore(self):
        pass
    
    def get_vectorstore(self, connection_args, embed_model):
        pass

    def ingest_to_vectorstore(self):
        pass
    
    # TODO: Add function to clear database
    # def clear_vectorstore():
    #     pass

class MilvusDBHandler(VectorDBHandler):
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL_NAME)
        self.vs = None
    
    def conenct_to_vectorstore(self, uri=MILVUS_LOCAL_URI, collection=MILVUS_COLLECTION):
        self.vs = MilvusVectorStore(
            embedding_function=self.embedder,
            connection_args={"uri": uri},
            collection_name=collection,
            primary_field="pk",   # <- match your collection schema
            auto_id=False,        # <- since we supply IDs below
        )

    def get_vectorstore(self, connection_args, embed_model):
        return self.vs

        # TODO: Decide if serialize helpers need to be moved to utils
    def serialize_metadata_value(self, value: Any) -> Any:
        """Convert non-JSON serializable values to JSON serializable ones"""
        if isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, 'isoformat'):  # Other date-like objects
            return value.isoformat()
        elif isinstance(value, (set, frozenset)):
            return list(value)
        elif hasattr(value, '__dict__'):  # Custom objects
            return str(value)
        return value

    def serialize_metadata(self, metadata: dict) -> dict:
        """Recursively serialize metadata to be JSON compatible"""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                serialized[key] = self.serialize_metadata(value)
            elif isinstance(value, list):
                serialized[key] = [self.serialize_metadata_value(item) if isinstance(item, dict) 
                                else self.serialize_metadata_value(item) for item in value]
            else:
                serialized[key] = self.serialize_metadata_value(value)
        return serialized


    def ingest_to_vectorstore(self, chunks: List[Document]) -> MilvusVectorStore:
        # Ensure each chunk has a stable ID to use as PK
        ids = []
        for d in chunks:
            d.metadata = self.serialize_metadata(d.metadata)
            cid = d.metadata.get("chunk_id")
            if not cid:
                cid = f"{d.metadata.get('doc_id','')}-{d.metadata.get('page','')}-{uuid4().hex}"
                d.metadata["chunk_id"] = cid
            ids.append(cid)

        self.vs.add_documents(chunks, ids=ids)
        return self.vs
    
class ElasticSearchHandler(VectorDBHandler):

    def __init__(self):
        self.es = None
        self.index = None

    def get_vectorstore(self):
        return self.es
    
    def connect_to_vectorstore(self, uri=config.ES_LOCAL_URI, index=config.ES_INDEX):
        self.es = Elasticsearch([uri])
        # Create index if missing (simple mapping)
        self.index = index
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, ignore=400)
            
    def ingest_to_vectorstore(self, chunks: List[Document]):
        """
        Optional: Keep hybrid search. We only push text (including table summaries) to ES.
        """
        
        actions = []
        for doc in chunks:
            if doc.metadata.get("chunk_type") in ("text", "table_summary"):
                actions.append({
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": doc.metadata["chunk_id"],
                    "source": doc.metadata.get("source", ""),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "chunk_type": doc.metadata.get("chunk_type", ""),
                    "parent_table_id": doc.metadata.get("parent_table_id", ""),
                    "text": doc.page_content,
                })
        # Bulk index
        if actions:
            from elasticsearch.helpers import bulk
            bulk(self.es, actions)