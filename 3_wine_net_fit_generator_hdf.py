
"""In this example, the dataset is read batch-by-batch by a custom generator."""

import io
import random
import time

import pandas
from keras.models import Sequential
from keras.layers import Dense


def seconds_to_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02.0f}h{:02.0f}m{:02.1f}s".format(hours, minutes, seconds)


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
            'validation': idlist[split_at:]}


class DataGenerator:
    def __init__(self, inputfile, batch_size, **kwargs):
        self.inputfile = inputfile
        self.batch_size = batch_size

    def generate(self, labels, idlist):
        while 1:
            for batch in self._read_data_from_hdf(idlist):
                batch = pandas.merge(batch, labels, how='left', on=['id'])
                Y = batch['label']
                X = batch.drop(['id', 'label'], axis=1)
                yield (X, Y)    

    def _read_data_from_hdf(self, idlist):
        chunklist = [idlist[i:i + self.batch_size] for i in range(0, len(idlist), self.batch_size)]
        for chunk in chunklist:
            yield pandas.read_hdf(self.inputfile, key='data', where='id in {}'.format(chunk))



def current_time():
    return time.time()


def elapsed_time(start):
    return seconds_to_time(current_time() - start)


# Configuration.
config = {
    'inputfile': 'datasets/wine.h5',
    'batch_size': 10,
    'epochs': 10,
}


START = current_time()


# Datasets.
labels = pandas.read_hdf('datasets/wine.h5', 'labels')
partitions = get_partitions(labels['id'], validation_split=0.2)


# Generators.
training_generator = DataGenerator(**config).generate(labels, partitions['train'])
validation_generator = DataGenerator(**config).generate(labels, partitions['validation'])


# Create model
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit_generator(generator=training_generator,
                    steps_per_epoch=len(partitions['train']) // config['batch_size'],
                    validation_data=validation_generator,
                    validation_steps=len(partitions['validation']) // config['batch_size'],
                    epochs=config['epochs'])

print("Elapsed: {}".format(elapsed_time(START)))