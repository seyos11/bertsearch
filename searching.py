import os
from pprint import pprint

from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch
from bert_serving.client import BertClient
SEARCH_SIZE = 100
os.environ['INDEX_NAME'] = 'jobsearch'
print(os.environ['INDEX_NAME'])
INDEX_NAME = os.environ['INDEX_NAME']

bc = BertClient(ip='localhost', output_fmt='list')
client = Elasticsearch('localhost:9200')

query = input(">>> ")
query_vector = bc.encode([query])[0]

script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
            "params": {"query_vector": query_vector}
        }
    }
}

response = client.search(
    index=INDEX_NAME,
    body={
        "size": SEARCH_SIZE,
        "query": script_query,
        "_source": {"includes": ["sentenceId", "personaId", "text"]}
    }
)
print(query)


