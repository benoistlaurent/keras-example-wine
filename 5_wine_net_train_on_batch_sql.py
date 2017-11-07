
"""In this example, the dataset is read batch-by-batch by a custom generator."""

import os
import random
import sqlite3
import time

from keras.models import Sequential
from keras.layers import Dense
import keras.utils as kutils

import pandas
import numpy
import tensorflow


# Setup to get reproducible results.
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(42)
random.seed(17)
tensorflow.set_random_seed(123456)


class DataGenerator:
    def __init__(self, inputfile, batch_size, labels, idlist, **kwargs):
        self.inputfile = inputfile
        self.batch_size = batch_size
        self.labels = labels
        self.idlist = idlist
        self.con = sqlite3.connect(self.inputfile)
        self._gen = self.generate()

    def __del__(self):
        self.con.close()  # close database connection

    def number_of_batches(self):
        """Return the number of batches the generator will produce."""
        return round(len(self.idlist) / self.batch_size)

    def shuffle(self):
        """Shuffle internal id list."""
        random.shuffle(self.idlist)

    def next_batch(self):
        """Get the next data batch."""
        batch = next(self._gen)
        return batch

    def generate(self):
        """Return a batch generator."""
        while 1:
            for batch in self._read_data_from_sql():
                batch = pandas.merge(batch, self.labels, how='left', on=['id'])
                Y = batch['label']
                X = batch.drop(['id', 'label'], axis=1)
                yield (X.values, Y.values)

    def _read_data_from_sql(self):
        """Generate the batches by reading the input SQL file."""
        chunklist = [self.idlist[i:i + self.batch_size]
                     for i in range(0, len(self.idlist), self.batch_size)]
        for i, chunk in enumerate(chunklist):
            query = 'select * from data where id in {}'.format(tuple(chunk))
            df = pandas.read_sql(query, self.con)
            yield df


def seconds_to_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02.0f}h{:02.0f}m{:02.1f}s".format(hours, minutes, seconds)


def current_time():
    """Return the number of seconds since the epoch.

    Equivalent to `time.time()`.
    """
    return time.time()


def elapsed_time(start):
    """Return the number of seconds between `start` and the moment
    this function is called."""
    return current_time() - start


def elapsed_to_str(start):
    """Return a string "hh:mm:ss" reprensenting the time elapsed since
    the start time (given in seconds)."""
    return seconds_to_time(current_time() - start)


def get_partitions(idlist, validation_split, shuffle=True):
    """Get the ids which will be used for the training and the validation.

    Args:
        idlist (list[str]): list of available ids in the dataset.
        validation_split (float 0 < x < 1): Fraction of the data to use as
            held-out validation data.
        shuffle (bool): If true, shuffle ids.

    Returns:
        dict[str]->list[str]: a dictionnary with two keys: 'train' and
            'validation'. To each key correspond a list of ids.
    """
    if shuffle:
        random.seed(42)  # fix random seed to always get the same results
        idlist = random.sample(list(idlist), len(idlist))
    split_at = int(len(idlist) * (1 - validation_split))
    return {'train': idlist[:split_at],
            'test': idlist[split_at:]}


def read_labels(path):
    """Read the label file (SQL)."""
    con = sqlite3.connect(path)
    labels = pandas.read_sql('select * from labels', con)
    con.close()
    return labels


def _step(model, data_gen, step_type='train'):
    """Run a single training/validation step.

    Args:
        model (keras.models.Sequential) : the network model
        data_gen (DataGenerator): data generator

    Returns:
        (float, float): average loss and accuracy for the step.
    """
    callback = model.train_on_batch
    if step_type not in ('train', 'test'):
        raise ValueError("step_type must be chosen ('train', 'test')")
    elif step_type == 'test':
        callback = model.test_on_batch
    loss, acc = 0.0, 0.0
    nbatches = data_gen.number_of_batches()
    for i in range(nbatches):
        x, y = data_gen.next_batch()
        y = kutils.to_categorical(y, 2)  # One-hot encode the labels
        l, a = callback(x, y)
        loss += l
        acc += a
    return loss / nbatches, acc / nbatches


def training_step(model, data_gen):
    """Run a training step.

    Args:
        model (keras.models.Sequential) : the network model
        data_gen (DataGenerator): data generator

    Returns:
        (float, float): average loss and accuracy for the step.
    """
    return _step(model, data_gen, 'train')


def validation_step(model, data_gen):
    """Run a validation step.

    Args:
        model (keras.models.Sequential) : the network model
        data_gen (DataGenerator): data generator

    Returns:
        (float, float): average loss and accuracy for the step.
    """
    return _step(model, data_gen, 'test')


def save_history_to_csv(path, history):
    """Save history dictionary to CSV file."""
    df = pandas.DataFrame(history)
    df.insert(0, 'epoch', range(df.shape[0]))   # insert the epoch number as fisrt column
    df.to_csv(path, index=False)



start = current_time()

# Configuration.
conf = {
    'inputfile': 'datasets/wine.db',
    'batch_size': 10,
    'epochs': 10,
}

# Datasets.
labels = read_labels(conf['inputfile'])
partitions = get_partitions(labels['id'], validation_split=0.2)


# Data generators.
training_generator = DataGenerator(labels=labels, idlist=partitions['train'], **conf)
test_generator = DataGenerator(labels=labels, idlist=partitions['test'], **conf)


# Create model
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = {
    'train-loss': [], 'train-accuracy': [],
    'test-loss': [], 'test-accuracy': [],
}

for epoch in range(conf['epochs']):
    _start = current_time()

    # Shuffle training set.
    training_generator.shuffle()

    # Training.
    start_train = current_time()
    train_loss, train_acc = training_step(model, training_generator)
    history['train-loss'].append(train_loss)
    history['train-accuracy'].append(train_acc)
    elapsed_train = elapsed_time(start)

    # Validation.
    test_loss, test_acc = training_step(model, test_generator)
    history['test-loss'].append(train_loss)
    history['test-accuracy'].append(train_acc)

    print("***** epoch {} *****".format(epoch + 1), flush=True)
    print("       loss: {:.3f}        accuracy: {:.3f}".format(train_loss, train_acc), flush=True)
    print("  test loss: {:.3f}   test accuracy: {:.3f}".format(test_loss, test_acc), flush=True)
    print("  elapsed:", elapsed_to_str(_start), flush=True)


save_history_to_csv('history.csv', history)

print("Elapsed:", elapsed_to_str(start))
