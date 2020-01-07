"""
Example script to create elasticsearch documents.
"""
import argparse
import json

import pandas as pd
from bert_serving.client import BertClient
bc = BertClient(output_fmt='list')


def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'text': doc['text'],
        'title': doc['title'],
        'text_vector': emb
    }
"""def load_dataset_5(path):
    docs = []
    count= 0
    data = parse_data(path)
    for row in data:
        doc = {
            'personaId': count
            'sentenceId':
            'sentence': ' '.join(row['persona_info'])
        }"""
def load_dataset_4(path):
    docs = []
    count = 0
    data = parse_data(path)
    for row in data:
        doc = {
            'title': count,
            'text': ' '.join(row['persona_info'])
        }
        docs.append(doc)
        count += 1
    return docs
def parse_data_2(path):
    docs = []
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        sentence_counter = 0
        persona_counter = 0
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                persona_counter += 1
                data.append({'persona_info': [], 'dialog': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)
                doc = {
                    'personaId':persona_counter,
                    'sentenceId':sentence_counter,
                    'sentence':persona_info
                }
                docs.append(doc)

            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])
            sentence_counter += 1
        return data

def parse_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                data.append({'persona_info': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)

        return data
def load_dataset_3(path):
    docs = []
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        sentence_counter = 0
        persona_counter = 0
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                persona_counter += 1
                data.append({'persona_info': [], 'dialog': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)
                doc = {
                    'personaId':persona_counter,
                    'sentenceId':sentence_counter,
                    'sentence':persona_info
                }
                docs.append(doc)

            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])
            sentence_counter += 1
        return docs
def load_dataset_2(path):
    docs = []
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                data.append({'persona_info': [], 'dialog': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)
                doc = {
                    'title':count,
                    'text':persona_info
                }
                docs.append(doc)
            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])
            count = count + 1
    return docs
def load_dataset(path):
    docs = []
    df = pd.read_csv(path)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'title': series.Title,
            'text': series.Description
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc['text'] for doc in batch_docs])
        for emb in embeddings:
            yield emb


def main(args):
    docs = load_dataset_3(args.data)
    with open(args.save, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
