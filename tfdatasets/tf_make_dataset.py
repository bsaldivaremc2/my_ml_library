import tensorflow as tf
import pandas as pd
def df_to_dataset(dataframe, shuffle=True, batch_size=32,buffer_size=2048):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size*4)
    ds = ds.batch(batch_size)
    return ds
