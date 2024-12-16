# elastic_chatgpt
ChatGPT Summary with Elasticsearch as a Private Datastore

Combining the search power of Elasticsearch with the Question Answering power of GPT

1. Python interface accepts user questions
- Generate a hybrid search request for Elasticsearch
- BM25 match on the title field
- kNN search on the title-vector field
- Boost kNN search results to align scores
- Set size=1 to return only the top scored document
- optinally search Bing if your private dataset does not have the answer
2. Search request is sent to Elasticsearch
3. Documentation body and original url are returned to python
4. API call is made to OpenAI ChatCompletion
- Prompt: "answer this question <question> using only this document <body_content from top search result>"
5. Generated response is returned to python
6. Python adds on original documentation source url to generated response and prints it to the screen for the user
