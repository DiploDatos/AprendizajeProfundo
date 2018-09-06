"""Constructs a matrix with FastText word embeddings for a smaller vocabulary.

The FastText model for English weights 9GB and has more than 2.5M word in
its vocabulary. The vocabulary in the moview dataset that we use has 50K.
We can filter out the embeddings that are not present in the movie's
vocabulary, to reduce the size of the word vectors matrix to load.

In this script we show how to do it, in case you want to preprocess
the embeddings for a different dataset. Also, we are not handling correctly
the OOV words, you are welcome to correct this error.

IMPORTANT: you can do this only when you are experimenting and trying to
find the best model. When you train your final model, you have to use the
entire set of wordvectors to account for words never seen before.
"""

import argparse
import pickle

from gensim.models import KeyedVectors
from sklearn.datasets import load_files
from utils import FilteredFastText


def read_args():
    parser = argparse.ArgumentParser(description='Filter FastText embeddings.')
    parser.add_argument('--embeddings_filename', type=str,
                        help='Name of the file with the FastText .vec model.')
    parser.add_argument('--output_filename', type=str,
                        help='Name of the file to store the resulting'
                             'FilteredFastText object.')
    args = parser.parse_args()
    return args


def main():
    args = read_args()
    dataset = load_files('dataset/txt_sentoken', shuffle=False)

    vocabulary = set()
    for instance in dataset['data']:
        vocabulary.update(instance.split())
    vocabulary = [x.decode("utf-8")  for x in vocabulary]
    print("Reducing embedding to {} words.".format(len(vocabulary)))

    # This takes like 4GB of RAM
    en_model = KeyedVectors.load_word2vec_format(args.embeddings_filename)

    # How many words in our vocabulary are not present in the wordvectors?
    words_no_embedding = len(set(vocabulary).difference(
        set(en_model.vocab)))
    print("There are {} word without embedding.".format(words_no_embedding))

    # We create and save the filtered model
    filtered_en_model = FilteredFastText(vocabulary, en_model)

    with open(args.output_filename, 'wb') as model_file:
        pickle.dump(filtered_en_model, model_file)


if __name__ == '__main__':
    main()