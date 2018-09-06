# Exercise 1

import argparse
import pandas

from keras.models import Sequential
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


def read_args():
    parser = argparse.ArgumentParser(description='Exercise 1')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--num_units', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    args = parser.parse_args()

    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset():
    dataset = load_files('dataset/txt_sentoken', shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)

    print('Training samples {}, test_samples {}'.format(
        len(X_train), len(X_test)))

    # TODO 1: Apply the Tfidf vectorizer to create input matrix
    # ....

    return X_train, X_test, y_train, y_test


def main():
    args = read_args()
    X_train, X_test, y_train, y_test_orginal = load_dataset()

    # TODO 2: Convert the labels to categorical
    # ...

    # TODO 3: Build the Keras model
    model = Sequential()
    # Add all the layers

    # model.compile(...)


    # TODO 4: Fit the model
    # hitory = model.fit(batch_size=??, ...)

    # TODO 5: Evaluate the model, calculating the metrics.
    # Option 1: Use the model.evaluate() method. For this, the model must be
    # already compiled with the metrics.
    # performance = model.evaluate(X_test, y_test)

    # Option 2: Use the model.predict() method and calculate the metrics using
    # sklearn. We recommend this, because you can store the predictions if
    # you need more analysis later. Also, if you calculate the metrics on a
    # notebook, then you can compare multiple classifiers.
    # predictions = ...
    # performance = ...

    # TODO 6: Save the results.
    # ...

    # One way to store the predictions:
    results = pandas.DataFrame(y_test_orginal, columns=['true_label'])
    results.loc[:, 'predicted'] = predictions
    results.to_csv('predicitions_{}.csv'.format(args.experiment_name),
                   index=False)



if __name__ == '__main__':
    main()