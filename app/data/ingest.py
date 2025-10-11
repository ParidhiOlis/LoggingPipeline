import os
import time
from typing import List
import logging

import logging
from app.config import *
from app.utils.preprocessing import load_documents, chunk_documents, load_chunks_to_milvus, load_chunks_to_elasticsearch
from app.models.parser import *
from sqlalchemy import create_engine, MetaData, Table, text
from datetime import datetime
from typing import List, Any


logger = logging.getLogger(__name__)


class MilvusHandler():

    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.vs = MilvusVectorStore(
            embedding_function=self.embedder,
            connection_args={"uri": MILVUS_LOCAL_URI},
            collection_name=MILVUS_COLLECTION,
            primary_field="pk",   # <- match your collection schema
            auto_id=False,        # <- since we supply IDs below
        )

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

    # Upsert to Milvus (vector)
    def upsert_to_milvus(self, chunks: List[Document]) -> MilvusVectorStore:

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

class ElasticSearchHandler():

    def __init__(self):
        self.es = Elasticsearch([ES_LOCAL_URI])
        # Create index if missing (simple mapping)
        if not self.es.indices.exists(index=ES_INDEX):
            self.es.indices.create(index=ES_INDEX, ignore=400)

    def upsert_to_es_text_only(self, chunks: List[Document]):
        """
        Optional: Keep hybrid search. We only push text (including table summaries) to ES.
        """
        
        actions = []
        for doc in chunks:
            if doc.metadata.get("chunk_type") in ("text", "table_summary"):
                actions.append({
                    "_op_type": "index",
                    "_index": ES_INDEX,
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


def ingest_data_with_location(location: str) -> None:
    try:
        # milvus_handler = MilvusHandler()
        # es_handler = ElasticSearchHandler()

        llama_parser = LlamaParser()

        documents = llama_parser.load_documents_in_folder(location)
        for d in documents:
            print(d)
        # chunks = llama_parser.chunk_documents(documents)
        # print("chunk 0 - ", chunks[0])

        # Load chunks to milvus db
        # milvus_vectorstore = milvus_handler.upsert_to_milvus(chunks)
        # print(f"Loaded {len(chunks)} to Milvus DB")
        # print("milvus test - ", milvus_vectorstore.similarity_search("How do cutaneous receptors work?", k=5))

        # Load chunks to elasticsearch
        # es_retriever = es_handler.upsert_to_es_text_only(chunks)
        # print(f"Loaded {len(chunks)} to ElasticSearch")
        # print("es test - ", es_retriever.get_relevant_documents("How do cutaneous receptors work?"))

    except Exception as e:
        logger.error(f"Error loading documents to vector DB: {e}", exc_info=True)
        return
    
def ingest_data_with_db(database: str, username: str, password: str, url: str, port: int, table: str, schema: str, windowHr: int = 72) -> None:
    try:
        milvus_handler = MilvusHandler()
        es_handler = ElasticSearchHandler()

        llama_parser = LlamaParser()

        # get the file_names from postgres sql database
        db_url = f"postgresql+psycopg2://{username}:{password}@{url}:{str(port)}/{database}"
        engine = create_engine(db_url)
        metadata = MetaData()
        tablename = Table(table, metadata, schema=schema, autoload_with=engine)
        file_info = {}
        # Run query
        with engine.connect() as conn:
            query = text(f"SELECT * FROM {tablename};")
            # query = text(f"SELECT * FROM {tablename} WHERE updated_at >= NOW() - INTERVAL '{str(windowHr)} hours';")
            result = conn.execute(query)
            columns = result.keys()
            print("columns - ", columns)
            for row in result:
                print(row)
                row_dict = row._asdict()  # Convert to dictionary
                file_info[row.id] = {}
                print(row_dict)
                for col in columns:
                    # print(f"{col} - {row_dict[col]}")
                    file_info[row.id][col] = row_dict[col]
                    
        # print("\n\n", file_info[1])

        # parse all the files 
        documents = llama_parser.load_documents_with_files(file_info)
        for d in documents:
            print(d)
        chunks = llama_parser.chunk_documents(documents)
        # print("chunk 0 - ", chunks[0])

        # Load chunks to milvus db
        milvus_vectorstore = milvus_handler.upsert_to_milvus(chunks)
        print(f"Loaded {len(chunks)} to Milvus DB")
        # print("milvus test - ", milvus_vectorstore.similarity_search("How do cutaneous receptors work?", k=5))

        # Load chunks to elasticsearch
        es_retriever = es_handler.upsert_to_es_text_only(chunks)
        print(f"Loaded {len(chunks)} to ElasticSearch")
        # print("es test - ", es_retriever.get_relevant_documents("How do cutaneous receptors work?"))

    except Exception as e:
        logger.error(f"Error loading documents to vector DB: {e}", exc_info=True)
        return

if __name__ == "__main__":
    data_directory = DOC_LOCATION 
    # ingest_data_with_location(data_directory)
    database_info = {
        'database': 'metadata_db',
        'username': 'app_user',
        'password': 'supersecret',
        'url': 'localhost',
        'port': 5432,
        'table': 'documents',
        'schema':  'core'
    }
    ingest_data_with_db(database_info['database'], 
                        database_info['username'], 
                        database_info['password'], 
                        database_info['url'], 
                        database_info['port'], 
                        database_info['table'], 
                        database_info['schema'])



