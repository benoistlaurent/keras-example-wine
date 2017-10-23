
"""In this example, the dataset is read batch-by-batch by a custom generator."""

import io
import random
import time

import pandas
# from keras.models import Sequential
# from keras.layers import Dense


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
        idlist = random.sample(list(idlist), len(idlist))
    split_at = int(len(idlist) * (1 - validation_split))
    return {'train': idlist[:split_at],
            'validation': idlist[split_at:]}


class DataGenerator:
    def __init__(self, csv, batch_size, **kwargs):
        self.csv = csv
        self.batch_size = batch_size

    def generate(self, labels, idlist):
        while 1:
            for batch in self._read_data_from_csv(idlist):
                batch = pandas.merge(batch, labels, how='left', on=['ids'])
                Y = batch['label']
                X = batch.drop(['ids', 'label'], axis=1)
                yield (X, Y)
        
    def _read_data_from_csv(self, idlist):
        batch = []
        with open(self.csv, 'rt') as f:
            headers = next(f).rstrip().split(',')
            for line in f:
                identifier = int(line.split(',')[0])
                if identifier in idlist:
                    batch.append(line.rstrip())
                    if len(batch) == self.batch_size:
                        df = pandas.read_csv(io.StringIO('\n'.join(batch)), header=None, names=headers)
                        yield df
                        batch = []
        if batch:
            yield pandas.read_csv(io.StringIO('\n'.join(batch)), header=None, names=headers)


def current_time():
    return time.time()


def elapsed_time(start):
    return seconds_to_time(current_time() - start)


# Configuration.
config = {
    'csv': 'datasets/wine.csv',
    'batch_size': 10,
    'epochs': 10,
}


START = current_time()

# Datasets.
labels = pandas.read_csv('datasets/wine_labels.csv')

print(labels.head())
print(labels.sample(frac=1).head())





exit()
partitions = get_partitions(labels['ids'], validation_split=0.2)



print(partitions['train'][:10])
exit()






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