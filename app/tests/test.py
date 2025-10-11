import os
import time
from typing import List
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory
)
from app.config import *
from app.models.retriever import *
from app.models.generator import *
from app.models.query_handler import *
from app.utils.preprocessing import *
from app.tests.test_values import *
from app.run import *


logger = logging.getLogger(__name__)

class AppResources:
    retriever: object
    generator: Optional[object] = None
    # query_handler: Optional[object] = None

# Input sentence structure
class SentenceInput(BaseModel):
    text: str

# Input query structure
class QueryInput(BaseModel):
    query: str
    user_info: Dict

def query(input: SentenceInput):
    log.info("Checking the sentence...")
    log.info("Input - " + str(input))
    response_dict = check_sentence(input.text)
    log.info("Response - " + str(response_dict))
    response_dict["input"] = input.text
    return response_dict

def predict(input: QueryInput, resources):
    log.info("Starting the Prediction...")
    log.info("Input - " + str(input))
    qa_chain = get_user_chain(input.user_info, resources)
    response_dict, memory = get_query_results(input.query, qa_chain)
    log.info("Response - " + str(response_dict))
    response_dict["input"] = input.text
    return response_dict

user_chains = {}
def get_user_chain(user_info: Dict, resources: AppResources):
    user_id = user_info.get("username")
    if user_id not in user_chains:
        print(f"Creating new chain for user {user_id}")
        # summary_generator = deepcopy(resources.generator)
        # summary_generator.max_tokens = 200
        # TODO: Try with ConversationBufferWindowMemory/ ConversationTokenBufferMemory/  ConversationSummaryMemory
        
        # memory = ConversationSummaryMemory(
        #     llm=resources.generator,
        #     memory_key="chat_history",   # <-- memory injects into this
        #     return_messages=True,
        #     output_key="output",
        #     max_token_limit=500
        # )

        memory = ConversationBufferWindowMemory(
            k=5,  # keep only the last 5 exchanges
            memory_key="chat_history",   # where memory is injected
            return_messages=True,
            output_key="output"
        )
        
        qa_chain = setup_rag_pipeline_with_subqueries(user_info, resources.retriever, resources.generator, memory)
        
        user_chain_info = {}
        user_chain_info['chain'] = qa_chain
        user_chain_info['memory'] = memory
        user_chains[user_id] = user_chain_info

    return user_chains[user_id]

def main():

    data_directory = DOC_LOCATION 
    user_queries = sdm_test_queries
    user_raw_sentences = sdm_test_sentences
    test_user_info = TEST_USER_INFO
    
    # Load Query Filter
    query_handler = QueryHandler()

    # Load Retriever
    milvus_uri = MILVUS_LOCAL_URI
    milvus_collection = MILVUS_COLLECTION
    es_uri = ES_LOCAL_URI
    es_index = ES_INDEX
    rag_retriever = RAG_Retriever(milvus_uri, milvus_collection, es_uri, es_index)
    vectorstore = rag_retriever.vectorstore
    retriever = rag_retriever.retriever

    # Load Generator
    rag_generator = RAG_Generator()
    generator = rag_generator.llm

    # Setup RAG Pipeline
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a QA system. You MUST ONLY use the provided context to answer. If the answer is not in the context, reply exactly with: 'I don't know based on the provided sources.' Never use prior knowledge or make assumptions. Cite sources as [S#]."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    resources = AppResources()
    resources.generator = generator
    resources.retriever = retriever

    if USE_QUERY_DECOMPOSITION:
        # qa_chain = setup_rag_pipeline_with_subqueries(test_user_info, retriever, generator, prompt)
        qa_chain_info = get_user_chain(TEST_USER_INFO, resources)
        qa_chain, memory = qa_chain_info['chain'], qa_chain_info['memory']
    else:
        qa_chain = setup_rag_pipeline(retriever, generator, prompt)
    if not qa_chain:
        return

    # Get the intent and query
    sentence_times = []
    user_queries = []
    for input in user_raw_sentences:
        # query = query_handler.query_formatter(raw_sentence, [])
        # user_queries.append(query)
        start_time = time.time()
        response_dict, memory = check_sentence(input.get("text"), memory=memory)
        end_time  = time.time()
        sentence_times.append(end_time - start_time)
        print("\n\nresponse dict - ", response_dict)
        print("\n\n memory - ", memory)
        user_queries.append(response_dict)
        # if query is not None and len(query) > 0:
        #     user_queries.append(query)
    
    answers = []
    question_times = []
    for user_query in user_queries:
        current_answers = []
        # print("user_query - ", user_query)
        if isinstance(user_query.get('message'), dict):
            for q in user_query.get('message').get('final_queries'):      
                query= q['query']
                highlight = q['highlight']
                print("query - ", query)
                # response = qa_chain.invoke(query)
                input = {
                    "value": query,
                    "highlight": highlight
                }
                start_time = time.time()
                response, memory = get_query_results_test(input, qa_chain)
                end_time = time.time()
                question_times.append(end_time-start_time)
                print("\n\n ++++++++ response - ", response)
                if response:
                    print(f"\n\nQuestion - {query}\nAnswer - {response}")
                    result = {}
                    result['query'] = query
                    result['response'] = response
                    result['highlight'] = highlight
                    current_answers.append(result)
                        # if(result['sources']):
                        #     answers.append(result)
                        #     print("Answer Appended")
                    # print("\ncurrent answers - ", current_answers)

        answers.append(current_answers)

    print(sentence_times)
    print(question_times)
    # print("User queries - ", type(user_queries))

    # # # Query the Pipeline
    # answers = []
    # start_time = time.time()
    # for user_query in user_queries:
    #     current_answers = []
    #     for q in user_query['final_queries']:
    #         query= q['query']
    #         highlight = q['highlight']
    #         print("query - ", query)
    #         response = qa_chain.invoke(query)
    #         if response:
    #             print(f"\n\nQuestion - {query}\nAnswer - {response}")
    #             result = {}
    #             result['query'] = query
    #             result['response'] = response
    #             result['highlight'] = highlight
    #             current_answers.append(result)
    #                 # if(result['sources']):
    #                 #     answers.append(result)
    #                 #     print("Answer Appended")
    #     answers.append(current_answers)
    # end_time = time.time()

    # print("\n\nall answers - ")
    # for a in answers:
    #     print("\n\n", a)
    # # print("time taken - ", end_time-start_time)
    # # write results to output file
    # file_path = "app/tests/test_results.txt"
    # with open(file_path, "a") as f:
    #     f.write("=================\n\n")
    #     f.write(f"Query Decomposition - {USE_QUERY_DECOMPOSITION}\n\n")
    #     f.write(f"Retrieval+Generation Time: {end_time-start_time}\n\n")
    #     for item in answers:
    #         f.write(str(item) + "\n")


if __name__ == "__main__":
    main()