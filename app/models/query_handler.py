import logging
import torch
from app.config import *
from langchain_community.llms import VLLM, HuggingFacePipeline, LlamaCpp
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.pydantic_v1 import BaseModel, Field, conlist
from typing import List


logger = logging.getLogger(__name__)

class QueryHandler():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = self.load_llm()

    def load_llm(self):
        try:
            if(self.device == 'cuda'):
                # LLM - Using vLLM
                if VLLM_USE_LOCAL:
                    llm = VLLM(model=LLM_MODEL_NAME)
                else:
                    llm = VLLM(
                        model=LLM_MODEL_NAME,
                        trust_remote_code=True,
                        max_model_len=LLM_MAX_MODEL_LEN,
                        temperature=LLM_TEMP,
                        top_p=LLM_TOP_P,
                        max_tokens=LLM_MAX_TOKENS  # or max_tokens_to_sample depending on version
                    )
            else:
                # llm = LlamaCpp(
                #     model_path=LLM_LLAMA_CPP,
                #     temperature=LLM_TEMP,
                #     max_tokens=LLM_MAX_TOKENS,
                #     top_p=LLM_TOP_P,
                #     verbose=False,
                #     n_ctx=LLM_MAX_CONTEXT_LEN,  # Context length (adjust based on model and memory)
                #     n_threads=4,  # Number of CPU threads (adjust based on your Mac's cores)
                #     n_gpu_layers=0  # Set to 0 for CPU-only; use >0 for Metal on Apple Silicon
                # )

                llm = ChatOpenAI(
                    model="gpt-4o",     
                    temperature=0,       # for deterministic output
                    openai_api_key=OPENAI_API_KEY
                )

            return llm

        except Exception as e:
            logger.error(f"Error loading LLM: {e}", exc_info=True)
            return None
    
    def query_formatter(self, text, seed_queries, memory=None):
        # SYSTEM_PROMPT = f"""You are a query refiner and generator for an enterprise. You will be provided 1) A free-form TEXT written by a person (may include casual chatter) 2) A list of SEED_QUERIES which is a dictionary containing 1) QUERY: previous queries (questions the person might be intending to ask) and 2) HIGHLIGHT: highlighted keywords associated with the previous queries. 
        #  Your job is to produce a list of distinct FINAL_QUERIES that: Capture the different possible intents of the user in the TEXT, Are distinct and non-overlapping, Focus on set of information the person might reasonably want to know, Avoid personal/casual topics, Keep original queries only if they can be modified with meaningful, factual context; otherwise, discard them for any trivial or no modifications, Include one contiguous substring highlight from TEXT per query, EXCLUDE any query/question whose answer is already stated in the TEXT."""

        SYSTEM_PROMPT = f"""Identify all distinct information needs in the TEXT or SEED_QUERIES. You may also be given a CONTEXT based on the chat history.

Rules:
1. Each sentence or complete clause can have zero or more intents; treat phrases like "don't know" or "need" as requests for that information.
2. Keep SEED queries only if improved with text-grounded context; discard trivial or duplicate queries.
3. Exclude a query only if its full answer is verbatim in TEXT; questions themselves are not answers.
4. For each query, return a contiguous 1-4 word substring from TEXT as 'highlight' that captures the intent. 
   - If the query is a broad informational request (e.g., asking for general details like "Who are investors of Samsung?"), use the relevant question phrase (e.g., "samsung investors") as the highlight if no specific context is available. 
   - Examples: For "What is the weather?", highlight "the weather"; for "Who won the game?", highlight "won the game".
5. Avoid personal or casual topics (e.g., opinions, non-factual queries).
6. If the TEXT contains incomplete clauses (e.g., lacks a clear subject, object, or actionable intent like "I want" or "What are its advantages?"):
   - Evaluate each complete clause or sentence independently for intents.
   - Resolve pronouns (e.g., "it", "its", "they", "this") using CONTEXT or previous queries/answers when possible.
   - Example: If CONTEXT includes "What is batch normalization?", then "What are its advantages?" becomes "What are the advantages of batch normalization?".
   - Only if no valid antecedent can be inferred, return an empty final_queries list and specify 'incomplete sentence or no clear intent' in no_query_reason.
7. If a sentence or clause is complete but lacks specific context for meaningful queries beyond a broad request (e.g., "Who are investors of Samsung?" or "What is the capital of France?"), treat the broad question as a valid intent unless its answer is verbatim in TEXT.
8. For incomplete clauses that imply intent (e.g., "I want", "What are its advantages"), you must attempt pronoun resolution against CONTEXT or chat history. If resolved, treat as a complete query. If not resolvable, then exclude with 'incomplete'.
9. Always identify clear factual questions as intents, even if broad, unless explicitly answered in TEXT. Prioritize extraction over exclusion.

Output must match QueryResponse schema:
- final_queries: list of FinalQuery objects (query, status[new|modified], reason, previous, highlight)
- no_query_reason: string"""
        
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

        structured_llm = self.llm.with_structured_output(QueryResponse)

        chat_history = "None"
        if memory is not None:
            print("\n\n==== memory value - ", memory)
            chat_history = memory.load_memory_variables({}).get("chat_history")
        
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TEXT:\n{text}\n\nSEED_QUERIES:\n{seed_queries}\n\nCONTEXT:\n{chat_history}"}
        ]
        # print("\n\n******** text - ", text)
        print("\n\n**** msgs - ", msgs)
        response = structured_llm.invoke(msgs).dict()
        print("\n\nquery_handler response - ", response)
        # print("\n\n***** response - ", response)
         # ✅ Update memory with new turn
        # print("\n\nresponse - ", response)
        # print("\n\ntype text - ", type(text))
        if memory is not None:
            memory.save_context(
                {"input": text},
                {"output": str(response)}
            )
        return response, memory


    def query_decomposer(self, query, n=QUERY_DECOMPOSITION_N):
        class DecomposeOutput(BaseModel):
            queries: List[str] = Field(description="List of short retrieval queries")
        decomposer_llm = self.llm.with_structured_output(DecomposeOutput)
        out: DecomposeOutput = decomposer_llm.invoke(
            f"Produce {n} retrieval queries for: {query}"
        )
        return out.queries

    def intent_to_query(self, intent):
        prompt_template = """
You are an assistant that converts an intent into a clear, concise user query.

Rules:
- Rewrite the intent as a natural question or command the user might have asked.
- Keep the query polite, clear, and actionable.
- Do not include any extra commentary.
- Output only the query.

Examples:

Intent: "User may want to know about Egypt."
Query: "Give a description of Egypt."

Intent: "User may want to know the location of IT department"
Query: "Where is the IT department located?"

Intent: "User wants to schedule a meeting for tomorrow"
Query: "Schedule a meeting for tomorrow."

Now process the following intent:
Intent: "{intent}"
"""
        prompt = prompt_template.format(intent=intent)
        response = self.llm.invoke(prompt)
        print("query - ", response.content.strip())
        return response.content.strip()

    def query_checker(self, query, top_n=3):
        prompt_template = f"""
You are an intent inference system that analyzes a user's query to determine what information they might find useful, even if they don't explicitly ask for it. Your task is to identify the potential information need based on the query and return it as a string. If no clear information need can be inferred or the query does not suggest a need for specific information, return "None".

Instructions:

1. Analyze the user's query to infer what information might be relevant or useful.
2. If the query implies curiosity, a desire for details, or a need for clarification about a topic, location, or entity, describe the specific information need as a concise string.
3. If the query is a statement of fact, a greeting, or lacks context to infer a specific need, return "None".
4. Avoid making assumptions beyond what the query reasonably suggests.
5. Output only the inferred information need as a string or "None". Do not include additional explanations or commentary.

Examples:
Query: I don't know about Egypt
Output: User may want an overview of Egypt

Query: Where is the IT department?
Output: User may want the location of the IT department

Query: I am fine. Thanks.
Output: None

Query: I am studying about cutaneous receptors.
Output: User may want detailed information on cutaneous receptors

Query: The only way to get any response is to go to Amazon India office.
Output: User may want the office details for Amazon India.

Now process the next query.

Query: {query}
Output:
"""

        prompt = prompt_template.format(query=query)
        response = self.llm.invoke(prompt)
        intent = response.content.strip()
        print("\nintent - ", intent)
        # Enforce exactly None or string
        if "none" in intent.lower():
            return None
        query = self.intent_to_query(intent)
        # if USE_QUERY_DECOMPOSITION and QUERY_DECOMPOSITION_N > 1:
        #     query = self.query_decomposer(query[0])
        return query