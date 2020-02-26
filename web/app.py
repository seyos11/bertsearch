import os
from pprint import pprint

from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch
from bert_serving.client import BertClient
SEARCH_SIZE = 10
os.environ['INDEX_NAME'] = 'jobsearch'
print(os.environ['INDEX_NAME'])
INDEX_NAME = os.environ['INDEX_NAME']
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def analyzer():
    bc = BertClient(ip='bertserving', output_fmt='list')
    client = Elasticsearch('elasticsearch:9200')

    query = request.args.get('q')
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
    pprint(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
