import os
import time
from typing import List
import logging
import asyncio

from langchain_core.prompts import ChatPromptTemplate

from app.config import *
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

query_handler = QueryHandler()

def check_sentence(raw_sentence: str, memory=None):
    # Get the intent and query
    try:
        queries, memory = query_handler.query_formatter(raw_sentence, [], memory)
        # print("\n\nrun memory - ", memory)
        print("result queries - ", queries)
        if queries.get('final_queries') is None or len(queries.get('final_queries')) == 0:
            return {
                "status": status['SUCCESS'],
                "message": messages['NONE']
            }, memory
        return {
            "status": status["SUCCESS"],
            "message": queries
        }, memory
    except Exception as e:
        log.error(f"Checking Query Failed - {e}", exc_info=True)
        return {
              "status": status["FAIL"],
            #   "message": messages["NONE"],
              "message": e,
            }, None

def get_query_results_test(query: Dict, qa_chain):
        
    # # Query the Pipeline
    try:
        question = query['value']
        highlight = query['highlight']
        print("question - ", question)

        async def run_async():
            response, memory = await qa_chain.ainvoke(question)
            # response, memory = qa_chain.invoke(question)
            print("response - ", response)
            if response:
                # log.info(f"\n\nQuestion - {question}\nAnswer - {response}")
                if(response.get('sources')):
                    result = {}
                    result['question'] = question

                    # Removing [S#] from answer
                    # cleaned = re.sub(r"\[S\d+\]", "", response["answer"])   # remove [S#]
                    # cleaned = re.sub(r"\s+", " ", cleaned)    # collapse multiple spaces
                    # response["answer"] = cleaned.strip()
                    response["answer"] = re.sub(r"\s*\[S\d+\]", "", response["answer"]).strip()

                    result['response'] = response
                    result['highlight'] = highlight
                    return {
                            "status": status["SUCCESS"],
                            "message": result
                        }, memory
                    
                else:
                    log.info("Ignoring - No Sources!")            
            
            return {
                "status": status['SUCCESS'],
                "message": messages['NONE']
            }, memory

        return asyncio.run(run_async())
   
        
    except Exception as e:
        log.error(f"Retrieval Failed - {e}", exc_info=True)
        return {
              "status": status["FAIL"],
            #   "message": messages["NONE"],
              "message": e,
            }, None

def get_query_results_test_sync(query: Dict, qa_chain):
        
    # # Query the Pipeline
    try:
            question = query['value']
            highlight = query['highlight']
            print("question - ", question)


            response, memory = qa_chain.invoke(question)
            # response, memory = qa_chain.invoke(question)
            print("response - ", response)
            if response:
                # log.info(f"\n\nQuestion - {question}\nAnswer - {response}")
                if(response.get('sources')):
                    result = {}
                    result['question'] = question

                    # Removing [S#] from answer
                    # cleaned = re.sub(r"\[S\d+\]", "", response["answer"])   # remove [S#]
                    # cleaned = re.sub(r"\s+", " ", cleaned)    # collapse multiple spaces
                    # response["answer"] = cleaned.strip()
                    response["answer"] = re.sub(r"\s*\[S\d+\]", "", response["answer"]).strip()

                    result['response'] = response
                    result['highlight'] = highlight
                    return {
                            "status": status["SUCCESS"],
                            "message": result
                        }, memory
                    
                else:
                    log.info("Ignoring - No Sources!")            
            
            return {
                "status": status['SUCCESS'],
                "message": messages['NONE']
            }, memory

   
        
    except Exception as e:
        log.error(f"Retrieval Failed - {e}", exc_info=True)
        return {
              "status": status["FAIL"],
            #   "message": messages["NONE"],
              "message": e,
            }, None

def get_query_results(query: str, qa_chain):
        
    # # Query the Pipeline
    try:
        question = query
        print("question - ", question)
        async def run_async():
            start_time = time.time()
            response, memory = await qa_chain.ainvoke(question)
            end_time = time.time()
            print("Response time - ", end_time - start_time)
            print("response - ", response)
            if response:
                # log.info(f"\n\nQuestion - {question}\nAnswer - {response}")
                if(response.get('sources')):
                    result = {}
                    result['question'] = question

                    # Removing [S#] from answer
                    # cleaned = re.sub(r"\[S\d+\]", "", response["answer"])   # remove [S#]
                    # cleaned = re.sub(r"\s+", " ", cleaned)    # collapse multiple spaces
                    # response["answer"] = cleaned.strip()
                    response["answer"] = re.sub(r"\s*\[S\d+\]", "", response["answer"]).strip()

                    result['response'] = response 
                    # result['highlight'] = highlight

                    return {
                            "status": status["SUCCESS"],
                            "message": result
                        }, memory
                    
                else:
                    log.info("Ignoring - No Sources!")            
            
            return {
                "status": status['SUCCESS'],
                "message": messages['NONE']
            }, memory
        return asyncio.run(run_async())
    

    except Exception as e:
        log.error(f"Retrieval Failed - {e}", exc_info=True)
        return {
              "status": status["FAIL"],
            #   "message": messages["NONE"],
              "message": e,
            }, None
    
def get_query_results_sync(query: str, qa_chain):
        
    # # Query the Pipeline
    try:
            question = query
            print("question - ", question)
        
            start_time = time.time()
            response, memory = qa_chain.invoke(question)
            end_time = time.time()
            print("Response time - ", end_time - start_time)
            print("response - ", response)
            if response:
                # log.info(f"\n\nQuestion - {question}\nAnswer - {response}")
                if(response.get('sources')):
                    result = {}
                    result['question'] = question

                    # Removing [S#] from answer
                    # cleaned = re.sub(r"\[S\d+\]", "", response["answer"])   # remove [S#]
                    # cleaned = re.sub(r"\s+", " ", cleaned)    # collapse multiple spaces
                    # response["answer"] = cleaned.strip()
                    response["answer"] = re.sub(r"\s*\[S\d+\]", "", response["answer"]).strip()

                    result['response'] = response 
                    # result['highlight'] = highlight

                    return {
                            "status": status["SUCCESS"],
                            "message": result
                        }, memory
                    
                else:
                    log.info("Ignoring - No Sources!")            
            
            return {
                "status": status['SUCCESS'],
                "message": messages['NONE']
            }, memory
        
    

    except Exception as e:
        log.error(f"Retrieval Failed - {e}", exc_info=True)
        return {
              "status": status["FAIL"],
            #   "message": messages["NONE"],
              "message": e,
            }, None
    

# def run(raw_sentence: str, qa_chain):

#     # Get the intent and query
#     queries = query_handler.query_formatter(raw_sentence, [])
#     if queries.get('final_queries') is None or len(queries.get('final_queries')) == 0:
#          return {
#               "status": status['SUCCESS'],
#               "message": messages['NONE']
#          }
        
#     # # Query the Pipeline
#     try:
#         answers = []
#         for q in queries['final_queries']:
#             query= q['query']
#             highlight = q['highlight']
#             response = qa_chain.invoke(query)
#             if response:
#                 log.info(f"\n\nQuestion - {query}\nAnswer - {response}")
#                 if(response.get('sources')):
#                     result = {}
#                     result['query'] = query
#                     result['response'] = response
#                     result['highlight'] = highlight
#                     answers.append(result)
#                 else:
#                     log.info("Ignoring - No Sources!")

#         if(answers):
#             return {
#                     "status": status["SUCCESS"],
#                     "message": answers
#                 }
#         else:
#             return {
#               "status": status['SUCCESS'],
#               "message": messages['NONE']
#             }
        
#     except Exception as e:
#         log.error(f"Retrieval Failed - {e}")
#         return {
#               "status": status["FAIL"],
#             #   "message": messages["NONE"],
#               "message": e,
#             }

if __name__ == "__main__":
    run()