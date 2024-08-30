import json


def load_json(output_file):
    with open(output_file) as f:
        data = json.load(f)
    return data