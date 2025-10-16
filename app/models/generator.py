import os
import time
from typing import List
import logging
import torch
from langchain_community.llms import VLLM, HuggingFacePipeline, LlamaCpp
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from app.config import *

logger = logging.getLogger(__name__)

# --- Load LLM Model --- 

class RAG_Generator():
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
                    model="gpt-4",      # or "gpt-4", "gpt-3.5-turbo"
                    temperature=LLM_TEMP,       # for deterministic output
                    top_p=LLM_TOP_P,
                    api_key=OPENAI_API_KEY
                )

                # tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_auth_token=HF_TOKEN)
                # model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True, use_auth_token=HF_TOKEN)

                # pipe = pipeline(
                #     "text-generation",
                #     model=model,
                #     tokenizer=tokenizer,
                #     max_new_tokens=LLM_MAX_TOKENS,
                #     temperature=LLM_TEMP,
                #     do_sample=True,
                #     top_p=LLM_TOP_P,
                #     return_full_text=False,
                #     pad_token_id=tokenizer.eos_token_id
                # )

                # llm = HuggingFacePipeline(pipeline=pipe, model_id=LLM_MODEL_NAME)

            return llm

        except Exception as e:
            logger.error(f"Error loading LLM: {e}", exc_info=True)
            return None


def build_local_llm(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
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





