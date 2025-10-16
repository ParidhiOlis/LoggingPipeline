# parse_and_index.py
import os   
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Any

import nest_asyncio
from llama_parse import LlamaParse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import ElasticSearchBM25Retriever
from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI
from app.config import *
from app.models.generator import build_local_llm
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class DocumentParser:

    def __init__(self, llm=None):
        if llm is None:
            self.llm = build_local_llm() 
        else:
            self.llm = llm
        self.llamaParser =  LlamaParse(
            api_key=LLAMAINDEX_API_KEY,
            result_type="markdown",  # easy to split text & tables later
            verbose=True,
            num_workers=4,
            max_pages=None,          # set an int to cap pages if desired
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def to_text(self, x) -> str:
        try:
            from langchain_core.messages import BaseMessage
        except Exception:
            BaseMessage = tuple()  # no-op if not installed

        if isinstance(x, str):
            return x
        if BaseMessage and isinstance(x, BaseMessage):
            return x.content or ""
        # HF pipeline sometimes returns list[{"generated_text": "..."}]
        if isinstance(x, list) and x and isinstance(x[0], dict) and "generated_text" in x[0]:
            return x[0]["generated_text"]
        # LangChain Generation objects
        try:
            from langchain_core.outputs import Generation, ChatGeneration
            if isinstance(x, (Generation, ChatGeneration)):
                # Generation has .text; ChatGeneration may have .message.content
                return getattr(x, "text", getattr(getattr(x, "message", None), "content", "")) or ""
        except Exception:
            pass
        return str(x)

    def _create_document_from_text(
        self,
        text: str,
        source: str,
        page_no: int = 1,
        item_index: int = None
    ) -> Document:
        """
        Helper to create a Document from text content.
        
        Args:
            text: Content text
            source: Source identifier
            page_no: Page/item number
            item_index: Optional index for list items
        
        Returns:
            LangChain Document
        """
        # Clean/normalize text if needed
        content = self.to_text(text)
        
        metadata = {
            "source": source,
            "page": page_no,
        }
        
        if item_index is not None:
            metadata["item_index"] = item_index
        
        return Document(
            page_content=content,
            metadata=metadata
        )


    def _dict_to_text(self, data: Dict[str, Any]) -> str:
        """
        Convert dictionary to readable text format.
        
        Args:
            data: Dictionary to convert
        
        Returns:
            Formatted text string
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                # For nested structures, use simple JSON-like format
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)

    def format_documents_with_llamaparse(self, file_path: str) -> List[Document]:
        """
        Returns a list of LangChain Documents with Markdown content.
        Each doc has metadata including source and page numbers (when available).
        """
        nest_asyncio.apply()
        # llama-parse returns LlamaIndex-style Documents (with .text and .metadata)
        lp_docs = self.llamaParser.load_data(file_path)
        print("lp_docs type - ", type(lp_docs[0]))

        lc_docs: List[Document] = []

        # TODO : add file metadata from postgres db here
        for d in lp_docs:
            # LlamaParse docs expose `.text` for content and `.metadata` as a dict
            content_raw = getattr(d, "text", "") or getattr(d, "page_content", "")
            content = self.to_text(content_raw)
            meta = dict(getattr(d, "metadata", {}) or {})

            # Try common page keys; default to 1 if not present
            page_no = int(meta.get("page", meta.get("page_number", meta.get("page_index", 1))))

            lc_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        **meta,
                        "source": str(Path(file_path)),
                        "page": page_no,
                    },
                )
            )

        return lc_docs
    
    def parse_documents_from_dict(self, data_dict: Dict[str, Dict[str, Any]]) -> List[List[Document]]:
        """
        Parse documents from a nested dictionary where each key is a data_id
        and value contains metadata and content.
        
        Args:
            data_dict: Dictionary with structure:
                {
                    data_id: {
                        "id": fid,
                        "ingested_at": timestamp,
                        "owner_id": str,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                        "acl": acl_info,
                        "content": content_data,
                        "raw": raw_data,
                        "source_type": str
                    }
                }
        
        Returns:
            List of document lists, one list per data element
        """
        all_docs: List[List[Document]] = []
        
        for data_id, data_info in data_dict.items():
            print(f"Parsing data_id: {data_id}")
            
            # Format documents from the content/raw data
            docs = self.format_documents_from_data(data_id, data_info)
            
            # Add persistent doc_id for all documents from this data element
            file_id = str(uuid.uuid4())
            
            # Get ACL metadata - fix this line to match the expected format
            acl_dict = {
                "acl": data_info.get('acl', {}),
                "owner": data_info.get('owner_id', 'UNK')
            }
            acl_meta = self.to_acl_metadata(acl_dict)
            
            # Enrich each document with metadata
            for d in docs:
                d.metadata = {
                    **(d.metadata or {}),
                    "source": data_id,
                    "doc_id": file_id,
                    "data_metadata": {
                        "id": data_info.get("id"),
                        "ingested_at": data_info.get("ingested_at"),
                        "owner_id": data_info.get("owner_id"),
                        "created_at": data_info.get("created_at"),
                        "updated_at": data_info.get("updated_at"),
                        "source_type": data_info.get("source_type")
                    },
                    "acl": acl_meta
                }
            
            all_docs.append(docs)
        
        if all_docs:
            print("\n\nFirst document - ", all_docs[0][0] if all_docs[0] else None)
        
        return all_docs
    
    def format_documents_from_data(
        self, 
        data_id: str, 
        data_info: Dict[str, Any]
    ) -> List[Document]:
        """
        Format documents from dictionary data structure.
        Handles both 'content' and 'raw' fields.
        
        Args:
            data_id: Identifier for this data element
            data_info: Dictionary containing content, raw data, and metadata
        
        Returns:
            List of LangChain Documents
        """
        lc_docs: List[Document] = []
        
        # Get content from either 'content' or 'raw' field
        content_data = data_info.get('content') or data_info.get('raw')
        
        if not content_data:
            print(f"Warning: No content found for data_id {data_id}")
            return lc_docs
        
        # Handle different content types
        if isinstance(content_data, str):
            # Single text content
            lc_docs.append(self._create_document_from_text(
                content_data, 
                data_id, 
                page_no=1,
                item_index=0    #TODO: Remove item_index as this wont be required
            ))
        
        elif isinstance(content_data, list):
            # Multiple rows/items (like norm_rows)
            for idx, item in enumerate(content_data, start=1):
                if isinstance(item, dict):
                    # Convert dict to text representation
                    text_content = self._dict_to_text(item)
                else:
                    text_content = str(item)
                
                lc_docs.append(self._create_document_from_text(
                    text_content,
                    data_id,
                    page_no=idx,
                    item_index=idx
                ))
        
        elif isinstance(content_data, dict):
            # Single dict content
            text_content = self._dict_to_text(content_data)
            lc_docs.append(self._create_document_from_text(
                text_content,
                data_id,
                page_no=1
            ))
        
        else:
            # Fallback: convert to string
            text_content = str(content_data)
            lc_docs.append(self._create_document_from_text(
                text_content,
                data_id,
                page_no=1
            ))
        
        return lc_docs

    #=========Start: Table extraction and summarization =========#

    # Split markdown into text/table substrings
    def extract_tables(self, md: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Return list of (table_text, (start, end)) found in markdown."""
        TABLE_FENCE_RE = re.compile(r"```(?:table)?\s*\n.*?\n```", re.DOTALL)  # fenced code blocks labeled as tables
        PIPE_TABLE_RE  = re.compile(r"(?:^\|.+\|\s*\n)+^\|(?:\s*:?-+:?\s*\|)+\s*$", re.MULTILINE)  # github-style pipe tables
        spans = []
        for m in TABLE_FENCE_RE.finditer(md):
            spans.append((m.group(0), m.span()))
        for m in PIPE_TABLE_RE.finditer(md):
            spans.append((m.group(0), m.span()))
        # Sort by start index to make removal deterministic
        spans.sort(key=lambda x: x[1][0])
        return spans
    
    # Summarize each table and link to it
    def summarize_table(self, table_md: str, doc_name: str) -> str:
        prompt = (
            "You are helping with Retrieval-Augmented Generation.\n"
            "Summarize the following table in 3–6 bullet points with key metrics, trends, and outliers.\n"
            "Be faithful to the table; do not hallucinate columns that are not present.\n\n"
            f"Document: {doc_name}\n"
            "Table (Markdown):\n"
            f"{table_md}\n\n"
            "Summary:"
        )
        return self.to_text(self.llm.invoke(prompt))

    #=========End: Table extraction and summarization =========#

    def split_markdown(self, md: str) -> Dict[str, List[str]]:
        """
        Separate text blocks and table blocks.
        Returns {"text": [...], "tables": [...]}
        """
        tables = self.extract_tables(md)
        table_blocks = [t[0] for t in tables]

        # Remove tables to get text-only markdown
        text_only = []
        cursor = 0
        for _, (s, e) in tables:
            if cursor < s:
                text_only.append(md[cursor:s])
            cursor = e
        if cursor < len(md):
            text_only.append(md[cursor:])

        # Clean empties
        text_only = [t.strip() for t in text_only if t.strip()]
        table_blocks = [t.strip() for t in table_blocks if t.strip()]
        return {"text": text_only, "tables": table_blocks}

    #=========Start: Document Chunking =========#

    def build_chunks_from_markdown(self, docs: List[Document]) -> List[Document]:
        """
        Page-wise splitting
        From llama-parse per-page markdown docs -> create:
        - text chunks (type=text, with page)
        - table chunks (type=table, with page)
        - table summaries (type=table_summary, with page and parent_table_id)
        """

        out: List[Document] = []

        for d in docs:
            md = d.page_content or ""
            meta = dict(d.metadata or {})
            source = meta.get("source") or meta.get("file_path") or "unknown"
            doc_id = meta.get("doc_id") or str(uuid.uuid4())
            page_no = int(meta.get("page", 1))  # <- page awareness

            # Split THIS PAGE ONLY into text blocks and table blocks
            parts = self.split_markdown(md)
            
            # print("text parts - ", len(parts["text"]))
            # print("table parts - ", len(parts["tables"]))

            # --- Text chunks (page-aware) ---
            for block in parts["text"]:
                for chunk in self.text_splitter.split_text(block):
                    out.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **meta,
                                "doc_id": doc_id,
                                "source": source,
                                "page": page_no,                 # <- keep page
                                "chunk_id": str(uuid.uuid4()),
                                "chunk_type": "text",
                            },
                        )
                    )

            # --- Table chunks + summaries (page-aware) ---
            for table_md in parts["tables"]:
                table_id = str(uuid.uuid4())

                # raw table chunk
                out.append(
                    Document(
                        page_content=table_md,
                        metadata={
                            **meta,
                            "doc_id": doc_id,
                            "source": source,
                            "page": page_no,                 # <- keep page
                            "chunk_id": table_id,
                            "chunk_type": "table",
                            "format": "markdown",
                        },
                    )
                )

                # summarized view (links back to parent table on same page)
                summary = self.summarize_table(table_md, Path(source).name)
                out.append(
                    Document(
                        page_content=summary,
                        metadata={
                            **meta,
                            "doc_id": doc_id,
                            "source": source,
                            "page": page_no,                 # <- keep page
                            "chunk_id": str(uuid.uuid4()),
                            "chunk_type": "table_summary",
                            "parent_table_id": table_id,
                        },
                    )
                )

        print("\n\n****** out - ", out)
        return out
    
    def chunk_documents(self, docs):
        all_chunks: List[Document] = []
        for doc in docs:
            file_chunks = self.build_chunks_from_markdown(doc)
            all_chunks.extend(file_chunks)
        return all_chunks
    
    #=========End: Document Chunking =========#

    #=========Start: Add Metadata =========#

    # RBAC metadata
    def build_acl_keyset(self, owner: str, allow_users=None, allow_roles=None, allow_groups=None, deny=None):
        allow_users  = [u.lower() for u in (allow_users or [])]
        allow_roles  = [r.lower() for r in (allow_roles or [])]
        allow_groups = [g.lower() for g in (allow_groups or [])]
        deny         = [u.lower() for u in (deny or [])]
        owner        = (owner or "").lower()

        # Pipe-delimited tokens make Milvus LIKE filters simple and fast enough
        tokens = [f"|owner:{owner}|"] + \
                [f"|user:{u}|"  for u in allow_users] + \
                [f"|role:{r}|"  for r in allow_roles] + \
                [f"|group:{g}|" for g in allow_groups] + \
                [f"|deny:{u}|"  for u in deny]
        return "".join(tokens)
    
    def to_acl_metadata(self, raw_meta: dict) -> dict:
        acl = raw_meta.get("acl", {})  # {"allow_roles":[],"allow_groups":[],"allow_users":[],"deny":[]}
        owner = (raw_meta.get("owner") or "").lower()

        meta = {
            "owner": owner,
            "allow_users": [u.lower() for u in acl.get("allow_users", [])],
            "allow_roles": [r.lower() for r in acl.get("allow_roles", [])],
            "allow_groups": [g.lower() for g in acl.get("allow_groups", [])],
            "deny_users":   [u.lower() for u in acl.get("deny", [])],
        }
        acl_meta = self. build_acl_keyset(
            owner=owner,
            allow_users=meta["allow_users"],
            allow_roles=meta["allow_roles"],
            allow_groups=meta["allow_groups"],
            deny=meta["deny_users"],
        )
        return acl_meta
    #=========End: Add Metadata =========#

    #=========Start: Utility Functions =========#
    def parse_documents_with_rbac(self, file_info: List[str]):
        all_docs: List[Document] = []
        for fid in file_info:
            p = file_info[fid]['filename']
            print(f"Parsing {p} with LlamaParse…")

            docs = self.format_documents_with_llamaparse(str(p))
            # add a persistent doc_id on all pages from this file
            file_id = str(uuid.uuid4())
            # get the access information for the doc
            acl_meta = self.to_acl_metadata(file_info[fid]['acl'])
            for d in docs:
                d.metadata = {**(d.metadata or {}), 
                              "source": str(p), 
                              "doc_id": file_id,
                              "file_metadata": file_info[fid],
                              "acl": acl_meta
                              }
            all_docs.append(docs)
        print("\n\ndocument - ", all_docs[0])
        return all_docs
    
    #=========Start: Utility Functions =========#
