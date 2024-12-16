import os
import streamlit as st
import openai
import requests
import dotenv
from elasticsearch import Elasticsearch
from pathlib import Path
from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel
from elasticsearch import Elasticsearch
from elasticsearch.client import MlClient

import getpass

# This code is part of an Elastic Blog showing how to combine
# Elasticsearch's search relevancy power with 
# OpenAI's GPT's Question Answering power
# https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code


# Required Environment Variables
# openai_api - OpenAI API Key
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password


dotenv.load_dotenv()


openai.api_key = os.environ['openai_api']
model = "gpt-4-turbo-preview"


#bing key
bing_search_api_key = os.environ['bing_subkey']
bing_search_endpoint = os.environ['bing_endpoint'] + \
    "v7.0/search"


#bing query
def bing_search(query):
    # Construct a request
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt}
    headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

    # Call the API
    try:
        response = requests.get(bing_search_endpoint,
                                headers=headers, params=params)
        response.raise_for_status()
        json = response.json()
        return json["webPages"]["value"]

        # print("\nJSON Response:\n")
        # pprint(response.json())
    except Exception as ex:
        raise ex

question = st.text_input
results = bing_search(question)

results_prompts = [f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results]




# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    return es

# Search ElasticSearch index and return body and URL of the result
def search(query_text):
    cid = os.environ['cloud_id']
    cp = os.environ['cloud_pass']
    cu = os.environ['cloud_user']
    es = es_connect(cid, cu, cp)

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "ml.inference.title.predicted_value"
                }
            }]
        }
    }

    knn = {
        "field": "ml.inference.title.predicted_value",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = 'search-nfl-hof'
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-4-turbo-preview", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]


#gpt-3.5-turbo
#gpt-4-turbo-preview
def bing_gpt(prompt, model="gpt-4-turbo-preview", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]




st.title("NFL Hall of Fame GPT")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")

# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from NFL Hall of Fame Docs."
search_bing = st.checkbox("Search with Bing?", value=False, key="bing")



if not search_bing:
   if submit_button:
     resp, url = search(query)
     prompt = f"Answer this question: {query}\nUsing only the information from this Elastic Dataset: {resp}\nIf the answer is not contained in the supplied doc reply '{negResponse}' and nothing else"
     answer = chat_gpt(prompt)

     if negResponse in answer:
        st.write(f"ChatGPT: {answer.strip()}")
     else:
        st.write(f"ChatGPT: {answer.strip()}\n\nDocs: {url}")
else:
     submit_button
     #results = search(query)
     prompt = f"Answer this question {query} {results_prompts}"
     
     answer= bing_gpt(prompt)

     st.write({answer.strip()})
