# elastic_chatgpt
ChatGPT Summary with Elasticsearch as a Private Datastore

Combining the search power of Elasticsearch with the Question Answering power of GPT

Link to Knowledge Article:

Python interface accepts user questions
Generate a hybrid search request for Elasticsearch
BM25 match on the title field
kNN search on the title-vector field
Boost kNN search results to align scores
Set size=1 to return only the top scored document
optinally search Bing if your private dataset does not have the answer
Search request is sent to Elasticsearch
Documentation body and original url are returned to python
API call is made to OpenAI ChatCompletion
Prompt: "answer this question using only this document <body_content from top search result>"
Generated response is returned to python
Python adds on original documentation source url to generated response and prints it to the screen for the user
