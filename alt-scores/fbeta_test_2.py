import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.utils.np_utils import to_categorical
# from tensorflow import keras

# metric = tf.keras.metrics.FBetaScore(beta=2.0, threshold=0.5)

y_true = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0]], np.int32)

y_pred = np.array([[0.2, 0.6, 0.7],
                   [0.2, 0.6, 0.6],
                   [0.6, 0.8, 0.0]], np.float32)

# metric.update_state(y_true, y_pred)

# result = metric.result()

# def fbeta_homemade(y_true, y_pred):

# y_true_flat = y_true.flatten()
# y_pred_flat = y_pred.flatten()

# class CustomFbetaMetric(tf.keras.metrics.Metric):
#     def __init__(self, beta=1.0, threshold=0.5, name='custom_fbeta_score', **kwargs):
#         super(CustomFbetaMetric, self).__init__(name=name, **kwargs)
#         self.beta = beta
#         self.threshold = threshold
#         # self.num_classes = num_classes
#         self.fbeta_score = tf.keras.metrics.FBetaScore( beta=self.beta, threshold=self.threshold)

#     def discretize(self, y_true, y_pred):
#         # Using np.where to discretize y_pred and y_true
#         y_pred = tf.where(y_pred >= self.threshold, 0, 1)
#         y_true = tf.where(y_true >= self.threshold, 0, 1)
#         return y_true, y_pred

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Use numpy function for discretization
#         y_true, y_pred = self.discretize(y_true, y_pred)

#         # Update the state using the fbeta_score metric
#         self.fbeta_score.update_state(y_true, y_pred, sample_weight=sample_weight)

#     def result(self):
#         return self.fbeta_score.result()

#     def reset_states(self):
#         self.fbeta_score.reset_states()

class CustomFbetaMetric(tf.keras.metrics.Metric):
    def __init__(self, beta=1.0, threshold=0.5, name='custom_fbeta_score', **kwargs):
        super(CustomFbetaMetric, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.threshold = threshold
        # self.num_classes = num_classes
        # self.fbeta_score = tf.keras.metrics.FBetaScore( beta=self.beta, threshold=self.threshold)
        self.precision_metric = tf.keras.metrics.Precision()
        self.recall_metric = tf.keras.metrics.Recall()

    # def discretize(self, y_true, y_pred, sample_weight=None):
    #     # Using np.where to discretize y_pred and y_true

    #     return y_true, y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use numpy function for discretization
        # y_true, y_pred = self.discretize(y_true, y_pred)

        y_pred_disc = tf.where(y_pred >= self.threshold, 0, 1)
        y_true_disc = tf.where(y_true >= self.threshold, 0, 1)
        print(y_pred_disc, y_true_disc)
        self.precision_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)
        self.recall_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)

        # Update the state using the fbeta_score metric
        # self.fbeta_score.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        print(precision, recall)
        beta_squared = self.beta ** 2
        fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + tf.keras.backend.epsilon())
        return fbeta

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()


class CustomFbetaMetric2(tf.keras.metrics.Metric):
    def __init__(self, beta=1.0, threshold=0.5, name='custom2_fbeta_score', **kwargs):
        super(CustomFbetaMetric2, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.threshold = threshold
        # self.num_classes = num_classes
        self.fbeta_score = tf.keras.metrics.FBetaScore( beta=self.beta)
        # self.precision_metric = tf.keras.metrics.Precision()
        # self.recall_metric = tf.keras.metrics.Recall()

    # def discretize(self, y_true, y_pred, sample_weight=None):
    #     # Using np.where to discretize y_pred and y_true

    #     return y_true, y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use numpy function for discretization
        # y_true, y_pred = self.discretize(y_true, y_pred)

        y_pred_disc = tf.where(y_pred >= self.threshold, 0, 1)
        y_true_disc = tf.where(y_true >= self.threshold, 0, 1)

        # self.precision_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)
        # self.recall_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)

        # Update the state using the fbeta_score metric
        self.fbeta_score.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)

    def result(self):
        # precision = self.precision_metric.result()
        # recall = self.recall_metric.result()

        # beta_squared = self.beta ** 2
        # fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + tf.keras.backend.epsilon())
        fbeta = self.fbeta_score.result()
        return fbeta

    def reset_states(self):
        # self.precision_metric.reset_states()
        # self.recall_metric.reset_states()
        self.fbeta_score.reset_states()

# # y_true_new_flat = pd.cut(y_true_flat, bins=[0, 0.5, 1], labels=[1, 0], retbins=False)
# y_pred_new_flat = pd.cut(y_pred_flat, bins=[0, 0.5, 1], labels=[0,1], include_lowest=True, retbins=False)

# print(y_pred_new)
# y_true_desc = y_true_new_flat.reshape(y_true.shape)
# y_pred_desc = y_pred_new_flat.reshape(y_pred.shape)


    # metric = tf.keras.metrics.FBetaScore(beta=2.0, threshold=0.5)

    # metric.update_state(y_true_new, y_pred_new)

    # res = metric.result()

    # return y_pred_desc
# def discretize(y_true):
#     # Example: Equal-width binning with 3 bins
#     bins = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#     return tf.raw_ops.Bucketize(input=y_true, boundaries=bins)
# y_true = np.array([[0.2, 0.6, 0.7],
#                    [0.6, 0.8, 0.0],
#                    [0.2, 0.6, 0.6]
#                    ], np.float32)
# y_pred = np.array([[0.2, 0.6, 0.7],
#                    [0.2, 0.6, 0.6],
#                    [0.6, 0.8, 0.0]], np.float32)

metric = CustomFbetaMetric()
metric.update_state(y_true, y_pred)

result = metric.result()
print(result)

# print(fbeta_homemade(y_true, y_pred))
# print(result)
