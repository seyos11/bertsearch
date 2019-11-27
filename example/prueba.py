import argparse
import json

import pandas as pd

def load_dataset_4(path):
    docs = []
    count = 0
    data = parse_data(path)
    for row in data:
        doc = {
            'title': count,
            'text': ''.join(row['persona_info'])
        }
        docs.append(doc)
        count += 1
    return docs
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

if __name__ == '__main__':
    
    data = load_dataset_4('/home/victor/UNI/Master1/TFM/datasets/ConvAI2/train_self_revised.txt')