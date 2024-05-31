################################################################################################################
import os, re
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

import custom_prompts

import ast
from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schema import (
    DecodingMethod,
    TextGenerationParameters,
)

from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from trulens_eval import Tru
from trulens_eval import TruCustomApp
from trulens_eval import Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.langchain import Langchain
from trulens_eval.tru_custom_app import instrument
from trulens_eval.feedback import prompts

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import numpy as np

# bge tokenizer and the model
bgel_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
bgel_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
bgel_model.eval()

# colbert tokenizer and the model
colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
colbert_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.chains import HypotheticalDocumentEmbedder
import langchain
langchain.debug = False

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

# from llama_index.indices.postprocessor import LongLLMLinguaPostprocessor
# from rouge import Rouge

import warnings
warnings.filterwarnings("ignore")

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

load_dotenv()
################################################################################################################


def bam_model(model_id='mistralai/mixtral-8x7b-instruct-v0-1', decoding_method='greedy', max_new_tokens=1000, 
              min_new_tokens=1, temperature=0.5, top_k=50, top_p=1, repetition_penalty=1):

    if decoding_method == 'greedy':
        decoding_method = DecodingMethod.GREEDY
        parameters=TextGenerationParameters(
            decoding_method=decoding_method,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty
        )
    else:
        decoding_method = DecodingMethod.SAMPLE
        parameters=TextGenerationParameters(
            decoding_method=decoding_method,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

    llm = LangChainInterface(
        model_id=model_id,
        client=Client(credentials=Credentials.from_env()),
        parameters=parameters,
    )

    return llm


def watsonx_model(model_id="ibm-mistralai/mixtral-8x7b-instruct-v01-q", decoding_method='greedy', max_new_tokens=500, 
                  min_new_tokens=1, temperature=0.5, top_k=50, top_p=1, repetition_penalty=1):
    params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_K: top_k,
        GenParams.TOP_P: top_p,
        GenParams.REPETITION_PENALTY: repetition_penalty
    }
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url=ibm_cloud_url,
        project_id=project_id,
        params=params,
    )
    return watsonx_llm


def make_retriever(df, method='default', llm=None):
    # Initialie the embedding model
    model_name = "intfloat/e5-large-v2"
    model_kwargs = {'device': 'cpu'}

    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25)
    data = text_splitter.create_documents(df["context"].to_list(), metadatas=df[["question"]].to_dict(orient="records"))

    if method == 'hyde':
    # Setting up the hypothetical answer retrieval mechanism
        prompt = 'Please write a paragraph to answer the below question.\nQuestion: {QUESTION}\n'
        prompt_template = PromptTemplate.from_template(prompt)


        # Generate hypothetical document for query using zero shot LLM call, and then embed that using the embeddings model defined above.
        embeddings = HypotheticalDocumentEmbedder.from_llm(llm,
                                                        embeddings_model,
                                                        custom_prompt=prompt_template,
                                                        verbose = False
                                                        )
        
        
        vectorstore = FAISS.from_documents(data, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}, verbose = False)
        return retriever
    
    if method == 'parent_doc':
        texts = ["text1", "text2", "text3"]
        faiss = FAISS.from_texts(texts, embeddings_model)

        # Define the child and parent splitters
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=75)

        store = InMemoryStore()

        # Initialize the ParentDocumentRetriever with FAISS
        retriever = ParentDocumentRetriever(
            vectorstore=faiss,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        data = list()
        for i in range(len(df)):
            doc = Document(
                metadata={
                    "question": df['question'][i],
                },
                page_content=df['context'][i])
            data.append(doc)

        # Add documents to the retriever
        retriever.add_documents(data, ids=None)
        return retriever

    if method == 'rag_fusion':
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        def reciprocal_rank_fusion(results: list[list], k=60):
            """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                and an optional parameter k used in the RRF formula """
            
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}

            # Iterate through each list of ranked documents
            for docs in results:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]

            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results
        
        vectorstore = FAISS.from_documents(data, embeddings_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}, verbose = False)

        retriever = generate_queries | retriever.map() | reciprocal_rank_fusion
        return retriever

    else: # default
        vectorstore = FAISS.from_documents(data, embeddings_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}, verbose = False)
        return retriever, embeddings_model
        

def colbert_reranker(docs, query):

    # start = time.time()
    scores = []

    # Function to compute MaxSim
    def maxsim(query_embedding, document_embedding):
        # Expand dimensions for broadcasting
        # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
        # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
        expanded_query = query_embedding.unsqueeze(2)
        expanded_doc = document_embedding.unsqueeze(1)

        # Compute cosine similarity across the embedding dimension
        sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)

        # Take the maximum similarity for each query token (across all document tokens)
        # sim_matrix shape: [batch_size, query_length, doc_length]
        max_sim_scores, _ = torch.max(sim_matrix, dim=2)

        # Average these maximum scores across all query tokens
        avg_max_sim = torch.mean(max_sim_scores, dim=1)
        return avg_max_sim

    # Encode the query
    query_encoding = colbert_tokenizer(query, return_tensors='pt')
    query_embedding = colbert_model(**query_encoding).last_hidden_state.mean(dim=1)

    # Get score for each document
    for document in docs:
        document_encoding = colbert_tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
        document_embedding = colbert_model(**document_encoding).last_hidden_state

        # Calculate MaxSim score
        score = maxsim(query_embedding.unsqueeze(0), document_embedding)
        scores.append({
            "score": score.item(),
            "document": document.page_content,
        })

    # print(f"Took {time.time() - start} seconds to re-rank documents with ColBERT.")

    # Sort the scores by highest to lowest and print
    sorted_data = sorted(scores, key=lambda x: x['score'], reverse=True)

    return sorted_data

def bge_reranker(docs, query):

    pairs = list()
    for document in docs:
        pairs.append([query, document.page_content])

    with torch.no_grad():
        inputs = bgel_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = bgel_model(**inputs, return_dict=True).logits.view(-1, ).float()
        # print(scores)

    scrs = list()
    for sc, dc in zip(scores, pairs):
        scrs.append({
                "score": sc,
                "document": dc[1],
            }
        )
    
    # Sort the scores by highest to lowest and print
    sorted_data = sorted(scrs, key=lambda x: x['score'], reverse=True)

    return sorted_data

def prompt_generation(context, query):
    template = (
        "<s>"
        "[INST] \n"
        "Context: {context} \n"
        "- Take the context above and use that to answer questions in a detailed and professional way. \n"
        "- If you don't know the answer just say \"I don't know\".\n"
        "- Refrain from using any other knowledge other than the text provided.\n"
        "- Don't mention that you are answering from the text, impersonate as if this is coming from your knowledge\n"
        "- For the questions whose answer is not available in the provided context, just say \"I don't know\".\n"
        "Question: {query} \n"
        "[/INST] \n"
        "</s>\n"
        "Answer: "
    )

    qa_template = PromptTemplate.from_template(template)
    return qa_template.format(context=context, query=query)

def compression_retriever(method, retriever=None, embeddings=None, llm=None):
    
    if method == 'LLMChainExtractor':
        
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever
    
    elif method == 'EmbeddingsFilter':
        # embeddings = HuggingFaceBgeEmbeddings()
        # embeddings = embeddings
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=retriever
        )
        return compression_retriever

    elif method == 'DocumentCompressorPipeline':

        # embeddings = HuggingFaceBgeEmbeddings() #embeddings #
        splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
    
    else:  #LLMLingua

        # node_postprocessor = LongLLMLinguaPostprocessor(
        #     model_name='NousResearch/Llama-2-7b-hf', #'NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT',
        #     device_map='cpu',
        #     instruction_str="Given the context, please answer the final question",
        #     target_token=300,
        #     rank_method="longllmlingua",
        #     additional_compress_kwargs={
        #         "condition_compare": True,
        #         "condition_in_question": "after",
        #         "context_budget": "+100",
        #         "reorder_context": "sort",  # enable document reorder,
        #         "dynamic_context_compression_ratio": 0.3,
        #     },
        # )
        # return node_postprocessor
        return None

def compression_metric(docs, compressed_docs):
    original_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(docs)]))
    compressed_contexts_len = len("\n\n".join([d.page_content for i, d in enumerate(compressed_docs)]))

    print("Original context length:", original_contexts_len)
    print("Compressed context length:", compressed_contexts_len)
    print("Compressed Ratio:", f"{original_contexts_len/(compressed_contexts_len + 1e-5):.2f}x")

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def test(retriever, llm):
    # Testing a sample query from the dataset
    query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
    docs = retriever.get_relevant_documents(query,verbose=False)

    pretty_print_docs(docs)

    # Generate the LLM Response
    context = "\n".join([doc.page_content for doc in docs])

    prompt = prompt_generation(context, query)
    print(prompt)

    result = llm.invoke(prompt)
    print(f"Answer: {result}")


class Processor:
    def __init__(self, retriever, llm, method='default'):
        self._retriever = retriever
        self._llm = llm
        self._method = method
    
    @instrument
    def retrieve_chunks(self, query, num_chunks):
        # self._retriever.search_kwargs = {"k": num_chunks}
        # try:
        if self._method == 'rag_fusion':
            docs = self._retriever.invoke({"question": query})
            docs = [doc[0].page_content for doc in docs[:num_chunks]]
        else:
            docs = self._retriever.get_relevant_documents(query)
            docs = [doc.page_content for doc in docs[:num_chunks]]
        return docs
        # except Exception as e:
        #     docs = self._retriever.invoke({"question": query})
        #     docs = [doc[0].page_content for doc in docs[:num_chunks]]
        #     return docs
    
    @instrument
    def join_chunks(self, chunks):
        return "\n".join(chunks)
    
    @instrument
    def respond_to_query(self, query, num_chunks=3):
        chunks = self.retrieve_chunks(query, num_chunks=num_chunks)
        context = self.join_chunks(chunks)
        prompt = prompt_generation(context, query)
        retries_left = 3
        while True:
            try:
                answer = self._llm.invoke(prompt).strip()
                break
            except Exception as e:
                print("Error while generating answer", e)
                if retries_left>0:
                    retries_left -= 1
                    print("Retrying. Retries Remaining -", retries_left)
                else:
                    raise
        return answer
    

class IBMLangchain(Langchain):
    def _create_chat_completion(self, prompt = None, messages = None, **kwargs):
        if prompt is not None:
            # prompt += "\nANSWER:\n"
            prompt = f"[INST]\n{prompt}\n[/INST]"
            predict = self.endpoint.chain.invoke(prompt, **kwargs)
            predict = re.sub(r'Score: (\d+)/\d+', r'Score: \1', predict)

        elif messages is not None:
            prompt = messages[0]['content']
            # prompt += "\nANSWER:\n"
            prompt = f"[INST]\n{prompt}\n[/INST]"
            predict = self.endpoint.chain.invoke(prompt, **kwargs)
            predict = re.sub(r'Score: (\d+)/\d+', r'Score: \1', predict)

        else:
            raise ValueError("`prompt` or `messages` must be specified.")
        
        return predict
    
    def _groundedness_doc_in_out(self, premise: str, hypothesis: str) -> str:
        """
        An LLM prompt using the entire document for premise and entire statement
        document for hypothesis.

        Args:
            premise (str): A source document
            hypothesis (str): A statement to check

        Returns:
            str: An LLM response using a scorecard template
        """
        assert self.endpoint is not None, "Endpoint is not set."

        return self.endpoint.run_in_pace(
            func=self._create_chat_completion,
            prompt=str.format(custom_prompts.LLM_GROUNDEDNESS_FULL_SYSTEM,) +
            str.format(
                prompts.LLM_GROUNDEDNESS_FULL_PROMPT,
                premise=premise,
                hypothesis=hypothesis
            )
        )

def eval_metrices(langchain_provider):
    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = (
        Feedback(
            langchain_provider.qs_relevance_with_cot_reasons,
            name="Context Relevance"
        )
        .on_input()
        .on(Select.RecordCalls.retrieve_chunks.rets[:])
        .aggregate(np.mean)
    )

    # Define a groundedness feedback function
    grounded = Groundedness(groundedness_provider=langchain_provider)
    f_groundedness = (
        Feedback(
            grounded.groundedness_measure_with_cot_reasons,
            name="Groundedness"
        )
        .on(Select.RecordCalls.join_chunks.rets) # collect context chunks into a list
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(
        langchain_provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input().on_output()

    return f_groundedness, f_qa_relevance, f_qs_relevance


def main():
    squad_dataset = load_dataset('squad')
    # print(squad_dataset['train'][0])

    df = pd.DataFrame(squad_dataset['train']).sample(200).reset_index(drop=True)
    print(df.shape)

    ## preprocess df if need be. we need two string columns - context and question

    # llm initialization
    mixtral_llm = bam_model(model_id="mistralai/mixtral-8x7b-instruct-v0-1", repetition_penalty=1.1)
    granite_llm = bam_model(model_id="ibm/granite-13b-chat-V2", repetition_penalty=1.1)

    # retrievers
    retriever, embeddings_model = make_retriever(df, llm=mixtral_llm) # default
    RETRIEVERS = {
        "baseline": retriever,
        "HyDE": make_retriever(df, method='hyde', llm=mixtral_llm),
        "parent_child": make_retriever(df, method='parent_doc'),
        "rag_fusion": make_retriever(df, method='rag_fusion', llm=mixtral_llm),
        "cc_llmChainExtractor": compression_retriever(method='LLMChainExtractor', retriever=retriever, llm=granite_llm),
        "cc_embeddingsFilter": compression_retriever(method='EmbeddingsFilter', retriever=retriever, embeddings=embeddings_model),
        "cc_docCompressorPipeline": compression_retriever(method='DocumentCompressorPipeline', retriever=retriever, embeddings=embeddings_model),
        
        "HyDE_cc_llmChainExtractor": compression_retriever(method='LLMChainExtractor', retriever=make_retriever(df, method='hyde', llm=mixtral_llm), llm=granite_llm),
        "HyDE_cc_embeddingsFilter": compression_retriever(method='EmbeddingsFilter', retriever=make_retriever(df, method='hyde', llm=mixtral_llm), embeddings=embeddings_model),
        "HyDE_cc_docCompressorPipeline": compression_retriever(method='DocumentCompressorPipeline', retriever=make_retriever(df, method='hyde', llm=mixtral_llm), embeddings=embeddings_model),
        "parent_child_cc_llmChainExtractor": compression_retriever(method='LLMChainExtractor', retriever=make_retriever(df, method='parent_doc'), llm=granite_llm),
        "parent_child_cc_embeddingsFilter": compression_retriever(method='EmbeddingsFilter', retriever=make_retriever(df, method='parent_doc'), embeddings=embeddings_model),
        "parent_child_cc_docCompressorPipeline": compression_retriever(method='DocumentCompressorPipeline', retriever=make_retriever(df, method='parent_doc'), embeddings=embeddings_model),

        # "rag_fusion_cc_llmChainExtractor": compression_retriever(method='LLMChainExtractor', retriever=make_retriever(df, method='rag_fusion', llm=mixtral_llm), llm=granite_llm),
        # "rag_fusion_cc_embeddingsFilter": compression_retriever(method='EmbeddingsFilter', retriever=make_retriever(df, method='rag_fusion', llm=mixtral_llm), embeddings=embeddings_model),
        # "rag_fusion_cc_docCompressorPipeline": compression_retriever(method='DocumentCompressorPipeline', retriever=make_retriever(df, method='rag_fusion', llm=mixtral_llm), embeddings=embeddings_model),
    } 

    # evaluation
    eval_llm = bam_model(model_id="mistralai/mixtral-8x7b-instruct-v0-1", repetition_penalty=1.1)
    langchain_provider = IBMLangchain(chain=eval_llm)
    f_groundedness, f_qa_relevance, f_qs_relevance = eval_metrices(langchain_provider=langchain_provider)

    results_df = pd.DataFrame()
    # Loop over retrievers
    for retriever_name, retriever in RETRIEVERS.items():
        print(f"\nRunning Evaluation for Retriever: {retriever_name}")
        print('-'*100)

        tru = Tru()
        tru.reset_database()

        if retriever_name == 'rag_fusion':
            rag = Processor(retriever, mixtral_llm, method='rag_fusion')
        else:
            rag = Processor(retriever, mixtral_llm)

        tru_rag = TruCustomApp(rag,
            app_id = retriever_name + '_RAG_Pipeline',
            feedbacks = [f_qs_relevance, f_groundedness, f_qa_relevance])
        
        
        with tru_rag as recording:
            for query in tqdm(df["question"], total=len(df)):
                ans = rag.respond_to_query(query)

        interim_results_df = tru.get_leaderboard(app_ids=[])
        interim_results_df['app_id'] = [retriever_name + '_RAG_Pipeline']

        results_df = pd.concat([results_df, interim_results_df], ignore_index=True)
        print('\nintermediate results after {}: '.format(retriever_name + '_RAG_Pipeline'))
        print(results_df.to_markdown())
    
    results_df = results_df[['app_id', 'Answer Relevance', 'Groundedness', 'Context Relevance', 'latency', 'total_cost']]
    results_df = results_df.round(2)

    # results_df.to_excel('./data/evaluation_results.xlsx', index=False)

    print('\nFinal evaluation results: ')
    print(results_df.to_markdown())
    '''
|    |   Answer Relevance |   Groundedness |   Context Relevance |   latency |   total_cost | app_id                                |
|---:|-------------------:|---------------:|--------------------:|----------:|-------------:|:--------------------------------------|
|  0 |           0.972432 |       0.712085 |            0.467901 |     3.935 |            0 | baseline_RAG_Pipeline                 |
|  1 |           0.950754 |       0.685559 |            0.467508 |     9.76  |            0 | HyDE_RAG_Pipeline                     |
|  2 |           0.967005 |       0.781458 |            0.629573 |     4.72  |            0 | parent_child_RAG_Pipeline             |
|  3 |           0.963317 |       0.672197 |            0.432997 |     6.745 |            0 | rag_fusion_RAG_Pipeline               |
|  4 |           0.899497 |       0.643684 |            0.561953 |     9.755 |            0 | cc_llmChainExtractor_RAG_Pipeline     |
|  5 |           0.965657 |       0.694985 |            0.513706 |     4.245 |            0 | cc_embeddingsFilter_RAG_Pipeline      |
|  6 |           0.965816 |       0.619591 |            0.465646 |     4.4   |            0 | cc_docCompressorPipeline_RAG_Pipeline |
    
|    |   Groundedness |   Context Relevance |   Answer Relevance |   latency |   total_cost | app_id                                 |
|---:|---------------:|--------------------:|-------------------:|----------:|-------------:|:---------------------------------------|
|  0 |       0.61655  |             0.55404 |           0.90201  |    25.695 |            0 | HyDE_cc_llmChainExtractor_RAG_Pipeline |
|  1 |       0.673413 |             0.5561  |           0.959799 |    14.455 |            0 | HyDE_cc_embeddingsFilter_RAG_Pipeline  |

|    |   Groundedness |   Context Relevance |   Answer Relevance |   latency |   total_cost | app_id                                         |
|---:|---------------:|--------------------:|-------------------:|----------:|-------------:|:-----------------------------------------------|
|  0 |       0.634313 |            0.480782 |           0.926131 |    18.285 |            0 | HyDE_cc_docCompressorPipeline_RAG_Pipeline     |
|  1 |       0.613588 |            0.632741 |           0.909045 |    11.29  |            0 | parent_child_cc_llmChainExtractor_RAG_Pipeline |
|  2 |       0.769566 |            0.812267 |           0.960804 |     6.985 |            0 | parent_child_cc_embeddingsFilter_RAG_Pipeline  |    

|    |   Groundedness |   Answer Relevance |   Context Relevance |   latency |   total_cost | app_id                                             |
|---:|---------------:|-------------------:|--------------------:|----------:|-------------:|:---------------------------------------------------|
|  0 |       0.670888 |           0.943147 |            0.524266 |     60.99 |            0 | parent_child_cc_docCompressorPipeline_RAG_Pipeline |

    '''


if __name__ == "__main__":
    main()
