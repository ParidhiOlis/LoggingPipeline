import logging
import torch
from app import config
from langchain_community.llms import VLLM, HuggingFacePipeline, LlamaCpp
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.pydantic_v1 import BaseModel, Field, conlist
from typing import List
from app.models.generator import build_local_llm


logger = logging.getLogger(__name__)


class FinalQuery(BaseModel):
    query: str = Field(description="A single, concise natural-language query derived from the intent.")
    status: str = Field(description="Status of the query [modified,new]")
    reason: str = Field(description="Short reason for adding the query")
    previous: str = Field(description="Previous query if the status is modified, otherwise return empty string")
    # added_context: str = Field(description="A single, concise natural-language query derived from the intent.")
    highlight: str = Field(description="Contiguous words in the input TEXT that needs to be highlighted")
    # notes: str = Field(description="A single, concise natural-language query derived from the intent.")

class QueryResponse(BaseModel):
    final_queries: List[FinalQuery]
    no_query_reason: str = Field(description="Reason for not adding queries, if final_queries is empty. If not empty, return empty string")
    # discarded_inputs: List[dict]
    # assumptions: List[str]

class QueryHandler():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = build_local_llm()
    
    def query_formatter(self, text, seed_queries, memory=None):
        
        # --- System Prompt for Query Generation ---
#         SYSTEM_PROMPT = f"""Identify all distinct information needs in the TEXT or SEED_QUERIES. You may also be given a CONTEXT based on the chat history.

# Rules:
# 1. Each sentence or complete clause can have zero or more intents; treat phrases like "don't know" or "need" as requests for that information.
# 2. Keep SEED queries only if improved with text-grounded context; discard trivial or duplicate queries.
# 3. Exclude a query only if its full answer is verbatim in TEXT; questions themselves are not answers.
# 4. For each query, return a contiguous 1-4 word substring from TEXT as 'highlight' that captures the intent. 
#    - If the query is a broad informational request (e.g., asking for general details like "Who are investors of Samsung?"), use the relevant question phrase (e.g., "samsung investors") as the highlight if no specific context is available. 
#    - Examples: For "What is the weather?", highlight "the weather"; for "Who won the game?", highlight "won the game".
# 5. Avoid personal or casual topics (e.g., opinions, non-factual queries).
# 6. If the TEXT contains incomplete clauses (e.g., lacks a clear subject, object, or actionable intent like "I want" or "What are its advantages?"):
#    - Evaluate each complete clause or sentence independently for intents.
#    - Resolve pronouns (e.g., "it", "its", "they", "this") using CONTEXT or previous queries/answers when possible.
#    - Example: If CONTEXT includes "What is batch normalization?", then "What are its advantages?" becomes "What are the advantages of batch normalization?".
#    - Only if no valid antecedent can be inferred, return an empty final_queries list and specify 'incomplete sentence or no clear intent' in no_query_reason.
# 7. If a sentence or clause is complete but lacks specific context for meaningful queries beyond a broad request (e.g., "Who are investors of Samsung?" or "What is the capital of France?"), treat the broad question as a valid intent unless its answer is verbatim in TEXT.
# 8. For incomplete clauses that imply intent (e.g., "I want", "What are its advantages"), you must attempt pronoun resolution against CONTEXT or chat history. If resolved, treat as a complete query. If not resolvable, then exclude with 'incomplete'.
# 9. Always identify clear factual questions as intents, even if broad, unless explicitly answered in TEXT. Prioritize extraction over exclusion.

# Output must match QueryResponse schema:
# - final_queries: list of FinalQuery objects (query, status[new|modified], reason, previous, highlight)
# - no_query_reason: string"""
        
        SYSTEM_PROMPT = """Extract distinct information needs (queries) from TEXT or SEED_QUERIES, optionally using CONTEXT from chat history.

Rules:
1. Each complete clause in TEXT may yield zero or more queries; treat phrases like "don't know" or "need" as requests for information.
2. Keep SEED queries only if improved by text-grounded or context-grounded information; discard trivial or duplicate queries.
3. Exclude a query only if its full answer is verbatim in TEXT; questions themselves are not answers.
4. For each query, select a 1–4 word contiguous substring from TEXT as 'highlight'. For broad informational requests or context-resolved queries (e.g., "What is Samsung?" or "Who are investors?" with context), use a phrase like "samsung investors" if no text-grounded span fits. Examples: TEXT="What is the weather?", highlight="the weather"; TEXT="Who are investors?" with CONTEXT="What is Samsung?", highlight="samsung investors".
5. Avoid personal, casual, or non-factual topics (e.g., opinions).
6. Handle incomplete clauses or missing subjects:
   - Resolve pronouns ("it", "its", "they", "this") and missing subjects (e.g., "Who are investors?" implying an entity) using the latest relevant topic/entity from CONTEXT or TEXT.
   - Examples:
     - CONTEXT="What is batch normalization?", TEXT="What are its advantages?" → query="What are the advantages of batch normalization?", highlight="batch normalization".
     - CONTEXT="What is Samsung?", TEXT="Who are investors?" → query="Who are the investors of Samsung?", highlight="samsung investors".
   - If no clear antecedent or intent, return no queries with reason "incomplete clause or no clear intent".
7. Broad factual questions (e.g., "What is Samsung?" or context-resolved queries) are valid unless answered verbatim in TEXT.
8. Prioritize inclusion of clear factual intents, favoring context-based resolution over exclusion.
9. When using CONTEXT, prioritize the most recent relevant query or entity (e.g., "Samsung" from "What is Samsung?") to resolve ambiguities in TEXT.

Output must match QueryResponse schema:
- final_queries: list of {query, status[new|modified], reason, previous, highlight}
- no_query_reason: string"""

        structured_llm = self.llm.with_structured_output(QueryResponse)

        chat_history = "None"

        # Optionally use memory to get chat history
        if memory is not None:
            print("\n\n==== memory value - ", memory)
            chat_history = memory.load_memory_variables({}).get("chat_history")
        
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TEXT:\n{text}\n\nSEED_QUERIES:\n{seed_queries}\n\nCONTEXT:\n{chat_history}"}
        ]

        response = structured_llm.invoke(msgs).dict()
       
       # Optionally save to memory
        if memory is not None:
            memory.save_context(
                {"input": text},
                {"output": str(response)}
            )
            
        return response, memory