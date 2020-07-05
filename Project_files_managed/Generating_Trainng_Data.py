import json


# this file converts json format of traning data to spacy format
def convert_json_to_spacy(filepath):
    with open(filepath, 'rb') as fp:
        data = fp.readlines()
        training_data = []
        for record in data:
            entities = []
            read_record = json.loads(record)
            text = read_record['text']
            entities_record = read_record['labels']
            for start, end, label in entities_record:
                entities.append((start, end, label))
                training_data.append((text, {"entities": entities}))

    return training_data
