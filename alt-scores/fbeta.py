
import tensorflow as tf

############ EXPERIMENT #################
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
        # print(y_true.shape, y_pred.shape)

        self.precision_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)
        self.recall_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)

        # Update the state using the fbeta_score metric
        # self.fbeta_score.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()

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

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred_disc = tf.where(y_pred >= self.threshold, 0, 1)
        y_true_disc = tf.where(y_true >= self.threshold, 0, 1)
        y_pred_disc = tf.cast(y_pred_disc, tf.float32)
        y_true_disc = tf.cast(y_true_disc, tf.float32)

        # Update the state using the fbeta_score metric
        self.fbeta_score.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)

    def result(self):
        fbeta = self.fbeta_score.result()
        return fbeta

    def reset_states(self):
        self.fbeta_score.reset_states()
########################################################################################
