
import sqlite3
import pandas


# Read data.
df_white = pandas.read_csv('datasets/winequality-white.csv', sep=';')
df_red = pandas.read_csv('datasets/winequality-red.csv', sep=';')

# Create labels: white wines are 1, red wines are 0.
df_white['label'] = 1
df_red['label'] = 0

# Concatenate the dataframes.
df = pandas.concat([df_white, df_red], ignore_index=True)

# Shuffle the big data frame so that white and red wines are mixed
# (fix random seed to always get the same shuffling).
df = df.sample(frac=1, random_state=42)

# Create a column 'id' (which is actually the order in the table).
df.insert(0, 'id', range(df.shape[0]))

# A data frame that contains only the labels.
labels = df[['id', 'label']]

# Remove labels from original dataframe.
df.drop(['label'], axis=1, inplace=True)

# Save data and labels to hdf5.
# `format` set to 'table' and `data_columns` to True so that hdf5 file
# can be read using the `where` specification (e.g. `where="id in [1, 2, 3]"`).
df.to_hdf('datasets/wine.h5', 'data', mode='w', format='table', data_columns=True)
labels.to_hdf('datasets/wine.h5', 'labels', mode='a', format='table', data_columns=True)


# Save data to csv as well.
df.to_csv('datasets/wine.csv', index=False)


# Save to sqlite as well.
con = sqlite3.connect('datasets/wine.db')
df.to_sql('data', con, if_exists='replace', index=False)
labels.to_sql('labels', con, if_exists='replace', index=False)
con.close()
