import os
import time
import asyncio
import threading
import concurrent.futures
from typing import List, Dict, Tuple, Set
import logging
import glob
import uuid
import traceback
import json
from typing import List, Optional, Any
from collections import OrderedDict
from typing import Dict
import re
import redis
from datetime import datetime
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.schema import StrOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_community.retrievers import ElasticSearchBM25Retriever
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import lru_cache
from app import config
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory
)

logger = logging.getLogger(__name__)

# --------- Preprocessing helpers -----------

def format_docs_with_sources(docs):
    # add metadata based source formatting here
    formatted_texts = []
    source_map: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
  # S# -> filename
    key_for_filename: dict[str, str] = {}               # filename -> S#
    next_idx = 1

    for d in docs:
        file_path = d.metadata.get("source", f"Doc{next_idx}")      # complete file path
        file_name = os.path.basename(d.metadata.get("source", f"Doc{next_idx}"))     # get the base filename
        file_owner = d.metadata.get("file_metadata", {}).get("owner_id", {}) + "@olis-ai.com"
        last_updated_at_raw = d.metadata.get("file_metadata", {}).get("updated_at", {})
        last_updated_at = datetime.fromisoformat(last_updated_at_raw).strftime("%B %d, %Y at %I:%M%p")        # Format it as "Month Day, Year at HH:MMAM/PM"
        # OrderedDict([('S1', 'all-olis.txt'), ('S2', 'product.txt')])
        # assign stable key per unique filename
        if file_name not in key_for_filename:
            key_for_filename[file_name] = f"S{next_idx}"
            file_info = {}
            file_info['file_name'] = file_name
            file_info['file_path'] = file_path
            file_info['owner'] = file_owner
            file_info['updated_at'] = last_updated_at
            source_map[key_for_filename[file_name]] = file_info
            next_idx += 1
        s_key = key_for_filename[file_name]

        formatted_texts.append(f"{s_key} ({file_name}):\n{d.page_content}")

    return "\n\n".join(formatted_texts), source_map


def prepare_context(inputs):
    docs = inputs["retrieved_docs"]
    context, sources = format_docs_with_sources(docs)
    return {"question": inputs["question"], "context": context, "sources": sources}


def setup_rag_pipeline(retriever, generator, prompt):
    chain = (
        {
            "question": RunnablePassthrough(),
            "retrieved_docs": retriever
        }
        | RunnableLambda(prepare_context)
        | {"answer": prompt | generator | StrOutputParser(), 
           "sources": lambda x: x["sources"]}
        | RunnableLambda(coalesce_citation_keys)  
        | RunnableLambda(prune_sources_by_citations)
    )

    return chain


# --------- Start: RBAC Debug Helpers ----------

# pre-filter
def debug_collection_fields(vectorstore):
    """Quick debug function to check what fields exist"""
    try:
        collection = vectorstore.col  # Access underlying Milvus collection
        schema = collection.schema
        
        print("Available fields:")
        field_names = []
        for field in schema.fields:
            print(f"  - {field.name} ({field.dtype})")
            field_names.append(field.name)
        
        # Check sample document
        try:
            results = collection.query(expr="", limit=1, output_fields=["*"])
            if results:
                print("\nSample document keys:")
                sample = results[0]
                for key in sample.keys():
                    print(f"  - {key}: {type(sample[key])}")
                    if key in ['metadata', 'file_metadata'] and isinstance(sample[key], dict):
                        print(f"    Nested keys in {key}:")
                        for nested_key in sample[key].keys():
                            print(f"      - {nested_key}")
        except Exception as e:
            print(f"Could not get sample: {e}")
            
        return field_names
    except Exception as e:
        print(f"Error checking fields: {e}")
        return []

def check_document_conditions(document, user_info):
    """
    Check which conditions a specific document satisfies
    """
    user = user_info['username'].lower()
    roles = [r.lower() for r in user_info.get('current_roles', [])]
    groups = [g.lower() for g in user_info.get('current_groups', [])]
    
    file_metadata = document.metadata.get('file_metadata', {})
    acl = file_metadata.get('acl', {})
    
    print(f"\n--- Document: {document.metadata.get('chunk_id', 'N/A')} ---")
    print(f"Owner: {file_metadata.get('owner_id')}")
    print(f"User: {user}")
    print(f"User Roles: {roles}")
    print(f"User Groups: {groups}")
    print(f"ACL: {acl}")
    
    # Check each condition
    conditions_met = []
    
    # Owner check
    if file_metadata.get('owner_id', '').lower() == user:
        conditions_met.append("✅ Document owner matches user")
    
    # User explicit allow
    allow_users = [u.lower() for u in acl.get('allow_users', [])]
    if user in allow_users:
        conditions_met.append(f"✅ User '{user}' explicitly allowed")
    
    # All users allowed
    if '_all_' in allow_users:
        conditions_met.append("✅ All users allowed")
    
    # Role checks
    allow_roles = [r.lower() for r in acl.get('allow_roles', [])]
    matched_roles = [r for r in roles if r in allow_roles]
    if matched_roles:
        conditions_met.append(f"✅ Roles matched: {matched_roles}")
    
    # Group checks
    allow_groups = [g.lower() for g in acl.get('allow_groups', [])]
    matched_groups = [g for g in groups if g in allow_groups]
    if matched_groups:
        conditions_met.append(f"✅ Groups matched: {matched_groups}")
    
    # Deny check
    deny_users = [u.lower() for u in acl.get('deny_users', [])]
    if user in deny_users:
        conditions_met.append(f"❌ User '{user}' explicitly denied")
    
    print("Conditions satisfied:")
    for condition in conditions_met:
        print(f"  {condition}")
    
    return conditions_met

# --------- End: RBAC Debug Helpers ----------

# ---------- Start: RBAC Helpers (pre and post filter) ----------

def build_acl_expr(user: str, roles=None, groups=None, tenant_id: Optional[str] = None) -> str:
    user = (user or "").lower()
    roles = [r.lower() for r in (roles or [])]
    groups = [g.lower() for g in (groups or [])]
    
    # Correct syntax: acl is nested inside file_metadata
    allow_conditions = [
        f'file_metadata["owner_id"] == "{user}"',
        f'array_contains(file_metadata["acl"]["allow_users"], "{user}")',
        f'array_contains(file_metadata["acl"]["allow_users"], "_ALL_")',
    ]
    
    # Add role conditions
    if roles:
        for role in roles:
            allow_conditions.append(f'array_contains(file_metadata["acl"]["allow_roles"], "{role}")')
    
    # Add group conditions  
    if groups:
        for group in groups:
            allow_conditions.append(f'array_contains(file_metadata["acl"]["allow_groups"], "{group}")')
    
    # Combine with OR
    allow_expr = ' or '.join(allow_conditions)
    
    # Deny condition (if you have deny_users in acl)
    deny_expr = f'not array_contains(file_metadata["acl"]["deny_users"], "{user}")'
    
    # Combine allow and deny
    base_expr = f'(({allow_expr}) and {deny_expr})'
    
    # Add tenant filter if you have tenant info in file_metadata
    if tenant_id:
        base_expr = f'file_metadata["tenant_id"] == "{tenant_id.lower()}" and ({base_expr})'
    
    return base_expr

def build_elasticsearch_rbac_filter(user_info):
    """Build ElasticSearch equivalent of RBAC filter"""
    user = user_info['username'].lower()
    roles = [r.lower() for r in user_info.get('current_roles', [])]
    groups = [g.lower() for g in user_info.get('current_groups', [])]
    
    should_clauses = [
        {"term": {"owner": user}},
        {"wildcard": {"acl_keyset": f"*|user:{user}|*"}},
        {"wildcard": {"acl_keyset": "*|user:_ALL_|*"}},
    ]
    
    for role in roles:
        should_clauses.append({"wildcard": {"acl_keyset": f"*|role:{role}|*"}})
    
    for group in groups:
        should_clauses.append({"wildcard": {"acl_keyset": f"*|group:{group}|*"}})
    
    return {
        "bool": {
            "must": [
                {"bool": {"should": should_clauses}},
                {"bool": {"must_not": {"wildcard": {"acl_keyset": f"*|deny:{user}|*"}}}}
            ]
        }
    }

def apply_rbac_to_retriever(retriever, user_info):
    """Apply appropriate RBAC based on retriever type"""
    if hasattr(retriever, 'vectorstore') and 'Milvus' in str(type(retriever.vectorstore)):
        # Milvus RBAC via expr
        expr = build_acl_expr(
            user_info['username'],
            user_info['current_roles'],
            user_info['current_groups'],
            # user_info.get('current_tenant')
        )
        retriever.search_kwargs["expr"] = expr
        
    elif 'ElasticSearch' in str(type(retriever)):
        # ElasticSearch RBAC via query filters
        rbac_filter = build_elasticsearch_rbac_filter(user_info)
        if hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs["filter"] = rbac_filter


# post-filter
def is_allowed(meta: Dict[str, Any], user: str, roles: List[str], groups: List[str]) -> bool:
    """RBAC: owner or explicitly allowed by user/role/group, and not denied."""
    file_metadata = meta.get("file_metadata")
    acl_metadata = file_metadata.get("acl")
    user = (user or "").lower()
    roles = {r.lower() for r in (roles or [])}
    groups = {g.lower() for g in (groups or [])}


    owner        = (file_metadata.get("owner_id") or "").lower()
    allow_users  = {u.lower() for u in (acl_metadata.get("allow_users")  or [])}
    allow_roles  = {r.lower() for r in (acl_metadata.get("allow_roles")  or [])}
    allow_groups = {g.lower() for g in (acl_metadata.get("allow_groups") or [])}
    deny_users   = {u.lower() for u in (acl_metadata.get("deny_users")   or meta.get("deny") or [])}

    if user in deny_users:
        return False
    if user == owner:
        return True
    if user in allow_users or '_ALL_'.lower() in allow_users:
        return True
    if roles & allow_roles:
        return True
    if groups & allow_groups:
        return True
    return False

def postfilter_factory(user_info):
    """
    Expects user_info to have .user (str), .roles (List[str]), .groups (List[str]).
    Works with outputs that contain either:
      - {"docs": List[Document], ...} OR
      - {"lists": List[List[Document]], ...}
    Returns the same shape with disallowed docs removed.
    """

    def _postfilter(payload: Dict[str, Any]) -> Dict[str, Any]:
    
        u, r, g = user_info['username'], user_info['current_roles'], user_info['current_groups']

        if "retrieved_docs" in payload and isinstance(payload["retrieved_docs"], list):
            payload["retrieved_docs"] = [d for d in payload["retrieved_docs"] if is_allowed(d.metadata or {}, u, r, g)]

        # If you also carry "sources" derived from docs and want to keep them in sync, recompute here.
        return payload
    return _postfilter

# ---------- End: RBAC Helpers (pre and post filter) ----------

# ---------- Start: Subqueries Handling and Answer Processing ----------

# Subquery decomposition 
class DecomposeOutput(BaseModel):
    queries: List[str] = Field(description="Short, diverse, retrieval-friendly subqueries")

class AnswerAndKeywords(BaseModel):
    answer: str = Field(
        description="Final answer text. Include inline source citations like [S1], [S2] that refer to the provided sources map."
    )
    # answer_span: str = Field(
    #     description="The key value to highlight (contiguous substring from 'answer'). "
    # )
    keywords: List[str] = Field(
        description="Relevant terms of the answer."
    )

def bold_terms_markdown(text: str, terms: list[str]) -> str:
    if not text or not terms:
        return text
    # sort longest-first to avoid partial overlaps
    patterns = [re.escape(t) for t in sorted({t for t in terms if t}, key=len, reverse=True)]
    if not patterns:
        return text
    pat = re.compile("|".join(patterns))

    # avoid code blocks
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    def wrap(m):
        s = m.group(0)
        return s if re.fullmatch(r"\*\*.+\*\*", s) else f"**{s}**"
    for i in range(0, len(parts), 2):
        parts[i] = pat.sub(wrap, parts[i])
    return "".join(parts)


def flatten_ak(payload):
    """payload: {'ak': AnswerAndKeywords, 'sources': ..., 'subqueries': ...}"""
    start_time = time.time()
    ak: AnswerAndKeywords = payload["answer"]
    answer = ak.answer
    keywords = [k for k in (ak.keywords or []) if k.strip()]
    answer_md = bold_terms_markdown(answer, keywords)
    # TODO: check keyword and markdown quality with other LLMs
    end_time = time.time()
    print("\n\n &&&&&& flatten time - ", end_time-start_time)
    return {
        "answer": answer,     # use answer instead of answer_md for raw answer
        # "answer_markdown": answer_md,
        # "keywords": keywords,   # Keywords would be highlighted in the amrkdown
        "sources": payload["sources"],
        "question": payload["question"],
        # "subqueries": payload.get("subqueries", [])   # Not including subqueries
    }

def timed_llm(llm, label="llm"):
    def _timed(inputs):
        start = time.time()
        result = llm.invoke(inputs)
        elapsed = time.time() - start
        print(f"\n\n &&&& {label} took {elapsed:.2f} seconds")
        return result
    return RunnableLambda(_timed)


def build_decomposer_chain(decompose_llm):
    parser = PydanticOutputParser(pydantic_object=DecomposeOutput)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Decompose the user's question into diverse, search-friendly sub-queries. "
             "Avoid duplicates; each under 15 words. Return ONLY valid JSON per schema.\n{format_instructions}"),
            ("human", "N={n}\nQuestion: {question}")
        ]
    )
    # chain = prompt | decompose_llm | parser
    chain = prompt | timed_llm(decompose_llm, label="decompose_llm") | parser
    fmt = parser.get_format_instructions()
    return chain, fmt

def _clean_subqueries(out: DecomposeOutput) -> List[str]:
    seen, cleaned = set(), []
    for q in out.queries:
        qn = q.strip()
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            cleaned.append(qn)
    return cleaned

# Retrieval over subqueries (batch)
def retrieve_for_subqueries_factory(user_info, retriever, per_query_k: int = 4):
    # add rbac and metadata based retrieval here
    def _run(inputs: Dict) -> Dict:
        subqueries: List[str] = inputs.get("subqueries")
        if not subqueries:  
            # Fallback: just use the question itself as one "subquery"
            subqueries = [inputs["question"]]

        # Handle different retriever configurations
        if hasattr(retriever, 'retrievers'):  # EnsembleRetriever
            retrievers_list = retriever.retrievers
        elif isinstance(retriever, list):
            retrievers_list = retriever
        else:
            retrievers_list = [retriever]

        # set k if retriever exposes search_kwargs
        for r in retrievers_list:
            # if hasattr(r, 'vectorstore') and 'Milvus' in str(type(r.vectorstore)):
            #     debug_collection_fields(r.vectorstore)
            #     break
            if hasattr(r, "search_kwargs"):
                r.search_kwargs["k"] = per_query_k      # set per query search results for milvus retriever
            if config.USE_RBAC_PREFILTER:
                # TODO: Add tenant_id filter functionality
                apply_rbac_to_retriever(r, user_info)

    
        try:
            lists: List[List[Document]] = retriever.batch(subqueries) if subqueries else []
        except Exception:
            lists = [retriever.invoke(q) for q in subqueries]

        # flatten + dedupe
        flat = [d for docs in lists for d in docs]
        def key(d: Document) -> Tuple:
            m = d.metadata or {}
            return (m.get("source"), m.get("page"), m.get("doc_id"), m.get("chunk_id"), (d.page_content or "")[:96])
        seen = set()
        unique: List[Document] = []
        for d in flat:
            k = key(d)
            if k not in seen:
                seen.add(k)
                unique.append(d)

        # if USE_RBAC_PREFILTER and unique:
        #     print("\n=== RBAC Condition Analysis ===")
        #     for doc in unique[:3]:  # Check first 3 documents
        #         check_document_conditions(doc, user_info)
        
        return {"question": inputs["question"], "retrieved_docs": unique, "subqueries": subqueries}

    return _run

# ---------- End: Subqueries Handling ----------

# ---------- Start: Answer post-processing----------

# Context prep wrapper (preserves subqueries)
def prepare_and_merge_factory(prepare_context_fn, memory = None):
    def _run(inputs: Dict) -> Dict:
        # Your prepare_context should accept a dict with keys: question, retrieved_docs
        prepared = prepare_context_fn(inputs)
        # Ensure prompt variables (adjust if your prompt uses a different key than 'context')
        # prepared must contain: {"question": ..., "context": ..., "sources": [...]}
        prepared["subqueries"] = inputs.get("subqueries", [])
        if(memory is None):
            prepared["chat_history"] = "None"
        else:
            prepared["chat_history"] = memory.load_memory_variables({}).get("chat_history")
        return prepared
    return _run

CITE_PATTERNS = [
    r"\[(S\d+)\]",          # [S1], [S2]
    r"\[(\d+)\]",           # [1], [2]
]

# Coalesce citations
def coalesce_citation_keys(payload: dict) -> dict:
    """
    If multiple S# map to the same filename, rewrite answer citations to the first (canonical) S#,
    and keep only canonical S# entries in sources (ordered by first appearance in the answer).
    """
    start_time = time.time()
    CITE_RE = re.compile(r"\[(S\d+)\]", flags=re.I)
    answer = payload.get("answer") or ""
    # print("\n\ninside coalesce payload sources - ", payload.get("sources"))
    sources: dict[str, str] = payload.get("sources") or {}

    # filename -> primary S#
    filename_to_primary: dict[str, str] = {}
    alias_to_primary: dict[str, str] = {}
    for s_key, fname in sources.items():
        primary = filename_to_primary.setdefault(fname['file_name'], s_key)
        if s_key != primary:
            alias_to_primary[s_key] = primary

    if not alias_to_primary:
        end_time_1 = time.time()
        return payload  # nothing to coalesce

    # 1) rewrite citations in the answer
    def repl(m):
        k = m.group(1).upper()
        return f"[{alias_to_primary.get(k, k)}]"
    answer = CITE_RE.sub(repl, answer)
    payload["answer"] = answer

    # 2) keep only canonical keys, ordered by first appearance in the (rewritten) answer
    cited_in_order = []
    seen = set()
    for m in CITE_RE.finditer(answer):
        k = m.group(1).upper()
        k = alias_to_primary.get(k, k)  # map to primary
        if k in sources and k not in seen:
            seen.add(k)
            cited_in_order.append(k)
    payload["sources"] = OrderedDict((k, sources[k]) for k in cited_in_order)
    end_time_2 = time.time()
    return payload

# Citation pruning
def _extract_cite_keys(answer_text: str, current_keys: set[str]) -> list[str]:
    """Return ordered unique keys (e.g., 'S1', 'S3') that appear in answer_text."""
    hits = []
    seen = set()

    # Build number->key map if your keys are like S1,S2,S3
    num_to_key = {}
    for k in current_keys:
        m = re.fullmatch(r"S(\d+)", k, flags=re.I)
        if m:
            num_to_key[int(m.group(1))] = k

    for pat in CITE_PATTERNS:
        for m in re.finditer(pat, answer_text, flags=re.I):
            token = m.group(1)  # 'S1' or '1'
            if token.upper().startswith("S"):      # [S1] style
                key = token.upper()
            else:                                  # [1] style
                # map number -> 'S#' if your keys are S#
                n = int(token)
                key = num_to_key.get(n, str(n))    # fallback to '1' if your keys are plain numbers
            if key in current_keys and key not in seen:
                seen.add(key)
                hits.append(key)

    return hits

# Prune sources to only those cited in the answer
def prune_sources_by_citations(payload: Dict) -> Dict:
    """
    Input payload: {"answer": str, "sources": Dict[str, Any], ...}
    Output: same, but sources filtered in the order cited.
    """
    start_time = time.time()
    answer = payload.get("answer", "") or ""
    sources: Dict[str, str] = payload.get("sources", {}) or {}
    cited_keys = _extract_cite_keys(answer, set(sources.keys()))
    if cited_keys:  # keep only cited, preserve order of first citation
        filtered = OrderedDict((k, sources[k]) for k in cited_keys)
    else:
        # If no citations found, you can either return empty or keep top-1.
        filtered = OrderedDict()   # or: OrderedDict(list(sources.items())[:1])

    payload["sources"] = filtered
    end_time = time.time()
    return payload
# ---------- End: Answer post-processing----------

# LLM timing wrapper
def time_llm(llm):
    def timed_invoke(input_data):
        start_time = time.time()
        result = llm.invoke(input_data)
        end_time = time.time()
        return result
    return timed_invoke


def setup_rag_pipeline_with_subqueries(user_info, retriever, generator, memory=None, prompt=None, decompose_generator=None, n_subqueries=config.QUERY_DECOMPOSITION_N, per_query_k=config.SUBQUERY_TOP_K):
    
    # ======= Memory Handler ========
    def memory_injector(inputs):
        """Injects memory state into the chain input."""
        history = memory.load_memory_variables({})
        inputs["chat_history"] = history.get("chat_history", "")
        return inputs

    def memory_updater(outputs):
        """Non-blocking memory update."""
        start_time = time.time()
        def update():
            memory.save_context(
                {"input": outputs.get("question", "")},
                {"output": outputs.get("answer", "")}
        )
        threading.Thread(target=update, daemon=True).start()
        end_time = time.time()
        return outputs, memory
    
    # A) subquery generator
    chat_history = memory.load_memory_variables({}).get("chat_history") or ""   # get chat history if available
    if prompt is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a QA system. You MUST ONLY use the provided context to answer. "
                    "If the answer is not in the context, reply exactly with: "
                    "'I don't know based on the provided sources.' Never use prior knowledge "
                    "or make assumptions. Cite sources as [S#]."),
            ("system", "Conversation so far:\n{chat_history}"),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

    if decompose_generator is None:
        decompose_generator = generator
    
    decomp_chain, fmt = build_decomposer_chain(decompose_generator)

    subquery_chain = (
        {"question": RunnablePassthrough(), "n": lambda _: n_subqueries, "format_instructions": lambda _: fmt}
        | decomp_chain
        | RunnableLambda(_clean_subqueries)  # -> List[str]
    )

    # B) retrieval over subqueries
    retrieve_fn = retrieve_for_subqueries_factory(user_info, retriever, per_query_k=per_query_k)

    # C) context prep (while keeping subqueries)
    prep = prepare_and_merge_factory(prepare_context, memory)

    # (D) structured answer + keywords
    answer_llm = generator.with_structured_output(AnswerAndKeywords)

    # E) Post-filter RBAC
    postfilter_node = RunnableLambda(postfilter_factory(user_info)) if config.USE_RBAC_POSTFILTER else None

    # F) final chain
    chain = {
            "question": RunnablePassthrough(),
            # "subqueries": subquery_chain,          # Uncomment to generate subqueries
        }
    chain = chain | RunnableLambda(retrieve_fn)             # 2) retrieve & dedupe
    if postfilter_node:
        chain = chain | RunnableLambda(postfilter_factory(user_info))
    chain = chain | RunnableLambda(prep) 
    chain = chain | {
            # "answer": (prompt | answer_llm),  # 4) answer
            "answer": (prompt | time_llm(answer_llm)),
            "question": lambda x: x["question"],
            "sources": lambda x: x["sources"],
            "subqueries": lambda x: x.get("subqueries", []),             # optional to return
        }
    chain = chain | RunnableLambda(flatten_ak)
    chain = chain | RunnableLambda(coalesce_citation_keys)  
    chain = chain | RunnableLambda(prune_sources_by_citations)
    
    # optional memory injection and update
    if config.USE_MEMORY_RETRIEVAL:
        chain = chain | RunnableLambda(memory_updater)

    return chain

