import os
import argparse
import json

import spacy
from spacy.tokens import DocBin

def read_tsv_file(input):
    result_list=[]
    with open(input) as input_file:
        for index, line in enumerate(input_file):

            # Skip the header
            if index > 0:
                # Split the dataset by "\t" character in order to extract the sentence and its label
                try:
                    sentence, label = line.split('\t')
                except Exception as e:
                    print(e)
                    print('index: {}'.format(index))
                    print('line: {}'.format(line))
                    ccc

                result_list.append(
                    {
                        "text": sentence,
                        "labels": [int(label)]
                    }
                )

    return result_list

def _read_categories(path_to_file):
    with open(path_to_file, 'rb') as input_file:
        categories_dict = json.load(input_file)

    return categories_dict

def convert_record(nlp, record, categories):
    # Get the label of the particular record
    label = record['labels'][0]

    # Extract the list of categories without the label of the particular record
    not_label_categories_dict = [category for key, category in categories.items() if int(key) != label]

    # Convert a record from the .tsv input file into a SpaCy Doc object
    doc = nlp.make_doc(record["text"])

    # All categories other than the true ones get value 0
    doc.cats = {category: 0 for category in not_label_categories_dict}

    # True labels get value 1
    doc.cats[categories[str(label)]] = 1

    return doc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='The path to the input dataset to convert to SpaCy\'s binary format.')
    parser.add_argument('-o', '--output', required=True,
                        help='The path to the input dataset to convert to SpaCy\'s binary format.')
    parser.add_argument('-c', '--categories', required=True,
                        help='The path to the .json file which contains the categories.')
    args = parser.parse_args()

    # Read the .json file which contains the list of categories
    categories_dict = _read_categories(path_to_file=args.categories)

    # Define an empty SpaCy pipeline for English language
    nlp = spacy.blank('en')

    # Read and parse the sentences with their labels
    records = read_tsv_file(args.input)

    # Convert the (sentence, label) pairs to SpaCy Doc object
    docs = [convert_record(nlp, record_dict, categories_dict) for record_dict in records]

    # Create the SpaCy's data structure that contains the SpaCy's Doc(s)
    doc_bin = DocBin(docs=docs)

    # Save it as .spacy file format
    doc_bin.to_disk(args.output)
    print('INFO: saved as .spacy binary format the {} [{} documents].'.format(
        args.input.split('/')[1].split('.')[0], len(docs)))

if __name__ == '__main__':
    main()