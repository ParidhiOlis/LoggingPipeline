# parse_and_index.py
import os
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

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
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LlamaParser:

    def __init__(self, llm=None):
        if llm is None:
            self.llm = self.build_local_llm() 
        else:
            self.llm = llm

    # Parse with LlamaParse
    def parse_with_llamaparse(self, file_path: str) -> List[Document]:
        """
        Returns a list of LangChain Documents with Markdown content.
        Each doc has metadata including source and page numbers (when available).
        """
        nest_asyncio.apply()
        parser = LlamaParse(
            api_key=LLAMAINDEX_API_KEY,
            result_type="markdown",  # easy to split text & tables later
            verbose=True,
            num_workers=4,
            max_pages=None,          # set an int to cap pages if desired
        )

        # llama-parse returns LlamaIndex-style Documents (with .text and .metadata)
        lp_docs = parser.load_data(file_path)
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

    def split_markdown_into_text_and_tables(self, md: str) -> Dict[str, List[str]]:
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

    # Summarize each table and link to it

    def build_local_llm(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        # tok = AutoTokenizer.from_pretrained(model_id)
        # mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        # gen = pipeline(
        #     "text-generation",
        #     model=mdl,
        #     tokenizer=tok,
        #     max_new_tokens=256,
        #     temperature=0.2,
        #     do_sample=False,
        #     return_full_text=False,
        #     pad_token_id=tok.eos_token_id,
        # )
        # return HuggingFacePipeline(pipeline=gen, model_id=model_id)

        llm = ChatOpenAI(
                        model="gpt-4o",      # or "gpt-4", "gpt-3.5-turbo"
                        temperature=LLM_TEMP,       # for deterministic output
                        top_p=LLM_TOP_P,
                        openai_api_key=OPENAI_API_KEY
                    )
        
        return llm

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


    # Build LangChain Documents for indexing
    def build_chunks_from_md(self, docs: List[Document]) -> List[Document]:
        """
        From llama-parse markdown docs -> create:
        - text chunks (type=text)
        - table chunks (type=table)
        - table summaries (type=table_summary, parent_table_id=<id>)
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )

        out: List[Document] = []

        for d in docs:
            md = d.page_content or ""
            meta = dict(d.metadata or {})
            source = meta.get("source") or meta.get("file_path") or "unknown"
            doc_id = meta.get("doc_id") or str(uuid.uuid4())

            # Split markdown into text blocks and table blocks
            parts = self.split_markdown_into_text_and_tables(md)

            # 4a) Text chunks
            for block in parts["text"]:
                for chunk in text_splitter.split_text(block):
                    out.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **meta,
                                "doc_id": doc_id,
                                "source": source,
                                "chunk_id": str(uuid.uuid4()),
                                "chunk_type": "text",
                            },
                        )
                    )

            # 4b) Table chunks + summaries
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
                            "chunk_id": table_id,
                            "chunk_type": "table",
                            "format": "markdown",
                        },
                    )
                )
                # summarized view (links back to parent table)
                summary = summarize_table(table_md, Path(source).name)
                print("summary - ", summary)
                out.append(
                    Document(
                        page_content=summary,
                        metadata={
                            **meta,
                            "doc_id": doc_id,
                            "source": source,
                            "chunk_id": str(uuid.uuid4()),
                            "chunk_type": "table_summary",
                            "parent_table_id": table_id,
                        },
                    )
                )

        return out


    def build_chunks_from_md_page(self, docs: List[Document]) -> List[Document]:
        """
        Page-wise splitting
        From llama-parse per-page markdown docs -> create:
        - text chunks (type=text, with page)
        - table chunks (type=table, with page)
        - table summaries (type=table_summary, with page and parent_table_id)
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )

        out: List[Document] = []

        for d in docs:
            md = d.page_content or ""
            meta = dict(d.metadata or {})
            source = meta.get("source") or meta.get("file_path") or "unknown"
            doc_id = meta.get("doc_id") or str(uuid.uuid4())
            page_no = int(meta.get("page", 1))  # <- page awareness

            # Split THIS PAGE ONLY into text blocks and table blocks
            parts = self.split_markdown_into_text_and_tables(md)
            
            # print("text parts - ", len(parts["text"]))
            # print("table parts - ", len(parts["tables"]))

            # --- Text chunks (page-aware) ---
            for block in parts["text"]:
                for chunk in text_splitter.split_text(block):
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

    # TODO Try separating the RBAC logic in a different module. Make the code more compact
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

    def load_documents_in_folder(self, folder: str = DOC_LOCATION):
        all_docs: List[Document] = []
        for p in Path(folder).rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt", ".md", ".png", ".jpg", ".jpeg"}:
                print(f"Parsing {p} with LlamaParse…")
                docs = self.parse_with_llamaparse(str(p))
                # add a persistent doc_id on all pages from this file
                file_id = str(uuid.uuid4())
                for d in docs:
                    d.metadata = {**(d.metadata or {}), "source": str(p), "doc_id": file_id}
                all_docs.append(docs)
        return all_docs
    
    def load_documents_with_files(self, file_info: List[str]):
        all_docs: List[Document] = []
        for fid in file_info:
            p = file_info[fid]['filename']
            print(f"Parsing {p} with LlamaParse…")
            docs = self.parse_with_llamaparse(str(p))
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
        return all_docs

    def chunk_documents(self, docs):
        all_chunks: List[Document] = []
        for doc in docs:
            file_chunks = self.build_chunks_from_md_page(doc)
            all_chunks.extend(file_chunks)
        return all_chunks

class NormalParser:
    
    def __init__(self):
        pass

    def load_documents(directory: str) -> List[Document]:
        """
        Loads documents from a directory, handling different file types including
        scanned documents and images using Tesseract OCR.
        """
        print("Loading documents...")
        start_time = time.time()
        print(f"Loading documents from directory: {directory}")
        
        PDF_LOADERS = [
            UnstructuredPDFLoader,
            PyPDFLoader,
            PDFMinerLoader,
        ]

        # Static mapping for non-PDF and non-image types
        loaders = {
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pptx": UnstructuredPowerPointLoader,
        }
        
        # Supported image extensions for OCR
        image_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
        
        # Gather all files regardless of extension
        all_files = []
        for ext in [*loaders.keys(), ".pdf", *image_extensions]:
            pattern = os.path.join(directory, f"**/*{ext}")
            all_files.extend(glob.glob(pattern, recursive=True))
        
        documents: List[Document] = []
        
        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            
            # Handle PDFs (including scanned PDFs)
            if ext == ".pdf":
                try:
                    # First try UnstructuredPDFLoader for text-based PDFs
                    for Loader in PDF_LOADERS:
                        try:
                            loader = Loader(file_path)
                            docs = loader.load()
                            documents.extend(docs)
                            logger.info(f"[{Loader.__name__}] loaded {file_path} → {len(docs)} docs")
                            break
                        except Exception as e:
                            logger.warning(f"[{Loader.__name__}] failed on {file_path}: {e}")
                    else:
                        # Fallback to OCR for scanned PDFs
                        logger.info(f"Attempting OCR on potentially scanned PDF: {file_path}")
                        try:
                            # Convert PDF to images
                            images = pdf2image.convert_from_path(file_path)
                            for i, image in enumerate(images):
                                text = pytesseract.image_to_string(image, lang="eng")
                                if text.strip():
                                    doc = Document(
                                        page_content=text,
                                        metadata={"source": file_path, "page": i + 1}
                                    )
                                    documents.append(doc)
                            logger.info(f"OCR processed {file_path} → {len(images)} pages")
                        except Exception as e:
                            logger.error(f"OCR failed for {file_path}: {e}")
                except Exception as e:
                    logger.error(f"All PDF loaders and OCR failed for {file_path}: {e}")
                    continue
            
            # Handle image files with OCR
            elif ext in image_extensions:
                try:
                    loader = UnstructuredImageLoader(file_path)
                    docs = loader.load()
                    if docs and docs[0].page_content.strip():
                        documents.extend(docs)
                        logger.info(f"[UnstructuredImageLoader] loaded {file_path} → {len(docs)} docs")
                    else:
                        # Fallback to Tesseract OCR if UnstructuredImageLoader fails
                        logger.info(f"UnstructuredImageLoader failed or empty, attempting Tesseract OCR on {file_path}")
                        image = Image.open(file_path)
                        text = pytesseract.image_to_string(image, lang="eng")
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={"source": file_path}
                            )
                            documents.append(doc)
                            logger.info(f"Tesseract OCR processed {file_path} → 1 doc")
                        else:
                            logger.warning(f"No text extracted from {file_path} via OCR")
                except Exception as e:
                    logger.error(f"Error processing image {file_path}: {e}", exc_info=True)
            
            # Handle non-PDF, non-image types
            else:
                loader_class = loaders.get(ext)
                if not loader_class:
                    print(f"Skipping unsupported extension {ext} for {file_path}")
                    continue
                try:
                    loader = loader_class(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {file_path} → {len(docs)} docs")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        print(f"{len(documents)} documents loaded in {elapsed:.2f} seconds")
        return documents
    
    # --- Chunk Documents ---
    def chunk_documents(self, documents: List) -> List:
        """Splits documents into smaller chunks."""
        print("Chunking documents....")
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) # Adjust chunk_size and chunk_overlap as needed
        chunks = text_splitter.split_documents(documents)

        elapsed = time.time() - start_time
        print(f"{len(chunks)} chunks created in {elapsed:.2f} seconds")
        return chunks
        
    
# ----------------------------------------
# Main entry: parse + index a folder
# ----------------------------------------
def index_folder(folder: str = DOC_LOCATION):
    all_chunks: List[Document] = []
    for p in Path(folder).rglob("*"):
        print("filename - ", p.name)
        if not p.is_file():
            continue
        # You can filter here if needed
        if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt", ".md", ".png", ".jpg", ".jpeg"}:
            print(f"Parsing {p} with LlamaParse…")
            docs = parse_with_llamaparse(str(p))
            # add a persistent doc_id on all pages from this file
            file_id = str(uuid.uuid4())
            for d in docs:
                # d.metadata = {**(d.metadata or {}), "source": str(p), "doc_id": file_id}
                meta = d.metadata or {}
                d.metadata = {
                    **meta,
                    "source": str(p),
                    "doc_id": file_id,
                    # LlamaParse usually sets this; default to 1 if missing
                    "page": meta.get("page", 1),
                }

            file_chunks = build_chunks_from_md_page(docs)
            all_chunks.extend(file_chunks)

