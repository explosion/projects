import os
import zipfile
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

def _unzip_file(path_to_input_file, path_to_output_directory):
    with zipfile.ZipFile(path_to_input_file, 'r') as zip_ref:
        zip_ref.extractall(path_to_output_directory)

def _extract_only_complete_sentence(df, with_label):
    # Extract only complete sentences
    subset_df = df.groupby(by='SentenceId').head(1)
    if with_label:
        subset_df = subset_df[['Phrase', 'Sentiment']]
    else:
        subset_df = pd.DataFrame(subset_df[['Phrase', 'PhraseId']])
        subset_df.columns = ['Phrase', 'PhraseId']

    subset_df.reset_index(drop=True, inplace=True)

    return subset_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='The path where there are all the downloaded files.')
    parser.add_argument('-o', '--output', required=True,
                        help='The path where to save the training set, validation set and test set files.')

    args = parser.parse_args()
    print('args: {}'.format(args))

    for file in os.listdir(args.input):
        if file.endswith('.tsv.zip'):
            print('file: {}'.format(file))
            _unzip_file(path_to_input_file='{}{}'.format(args.input, file),
                        path_to_output_directory=args.output)

    ### TRAINING SET ###

    training_set_df = pd.read_csv('{}train.tsv'.format(args.output), sep='\t', header=0)
    training_set_df = _extract_only_complete_sentence(df=training_set_df, with_label=True)

    # Split the training set into training set and validation set
    X, y = training_set_df.drop(columns='Sentiment'), training_set_df['Sentiment'].copy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, random_state=42, stratify=y)

    # Save to .csv files
    X_train.merge(y_train, left_index=True, right_index=True, how='inner').to_csv('{}{}'.format(args.input, 'training_set.tsv'),
                                                                                  sep='\t', index=False, header=True)
    X_val.merge(y_val, left_index=True, right_index=True, how='inner').to_csv('{}{}'.format(args.input, 'validation_set.tsv'),
                                                                              sep='\t', index=False, header=True)

    ### TEST SET ###
    y_test = pd.read_csv('{}test_set.csv'.format(args.input), sep=',', header=0)
    X_test = pd.read_csv('{}test.tsv'.format(args.output), sep='\t', header=0)
    X_test = _extract_only_complete_sentence(df=X_test, with_label=False)

    test_set_df = X_test.merge(y_test, how='inner', left_on='PhraseId', right_on='PhraseId')
    test_set_df.drop(columns='PhraseId', inplace=True)
    test_set_df.to_csv('{}{}'.format(args.input, 'test_set.tsv'), sep='\t', index=False, header=True)

    print('INFO: {} sentences will be used as training set.'.format(X_train.shape[0]))
    print('INFO: {} sentences will be used as validation set.'.format(X_val.shape[0]))
    print('INFO: {} sentences will be used as test set.'.format(X_test.shape[0]))

    os.remove('{}{}'.format(args.input, 'train.tsv'))
    os.remove('{}{}'.format(args.input, 'test.tsv'))
    os.remove('{}{}'.format(args.input, 'test_set.csv'))
    os.remove('{}{}'.format(args.input, 'training_set.tsv.zip'))
    os.remove('{}{}'.format(args.input, 'validation_set.tsv.zip'))

if __name__ == '__main__':
    main()