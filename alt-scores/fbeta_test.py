import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.utils.np_utils import to_categorical
# from tensorflow import keras

metric = tf.keras.metrics.FBetaScore(beta=2.0, threshold=0.5)

y_true = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0]], np.int32)

y_pred = np.array([[0.2, 0.6, 0.7],
                   [0.2, 0.6, 0.6],
                   [0.6, 0.8, 0.0]], np.float32)

metric.update_state(y_true, y_pred)

result = metric.result()


# def discretize(y_true):
#     # Example: Equal-width binning with 3 bins
#     bins = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#     return tf.raw_ops.Bucketize(input=y_true, boundaries=bins)

print(pd.cut([0.2, 0.6, 0.7], bins=[0, 0.5, 1], labels=[1, 0], retbins=False))

# print(result)
