import sklearn.datasets
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import sklearn
import sklearn.model_selection
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
# np.random.seed(0)
# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(1234)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

def generate_continuous_data(samples=1000, features=10, feature_range=(0, 1)):
    """
    Generate synthetic data with continuous features and continuous labels in the range [0,1].
    
    Args:
        samples (int): Number of samples to generate.
        features (int): Number of features for each sample.
        feature_range (tuple): Min and max value for the feature ranges.
    
    Returns:
        X (np.ndarray): Features matrix of shape (samples, features).
        y (np.ndarray): Continuous labels in the range [0, 1] of shape (samples, 1).
    """
    # Generate features with uniform distribution in the given range
    X = np.random.uniform(low=feature_range[0], high=feature_range[1], size=(samples, features))
    
    # Generate continuous labels based on a function of the features
    # Here we use a simple example: normalized sum of the features
    y = np.sum(X, axis=1)
    
    # Normalize labels to be in the range [0, 1]
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Reshape y to be of shape (samples, 1) to match model expectations
    y = y.reshape(-1, 1)
    
    return X, y

# Example usage:
X, y = generate_continuous_data(samples=10000, features=10, feature_range=(0, 100))
# X_test, y_test = generate_continuous_data(samples=1000, features=10, feature_range=(0, 100))

# print(y_train.shape, y_test.shape)

def generate_dataset(n_samples, n_features, skew_factor=5):
    """
    Generates a dataset with multiple features and a predictor constrained between 0 and 1.
    The predictor is heavily skewed towards values close to 1, controlled by skew_factor.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - n_features (int): Number of features in the dataset.
    - skew_factor (float): Controls how heavily the predictor is skewed towards 1.
                           Higher values will produce more skewed predictors.

    Returns:
    - pd.DataFrame: A DataFrame with 'n_features' feature columns and a 'predictor' column.
    """

    # Generate random features (independent variables)
    features = np.random.randn(n_samples, n_features)
    
    # Generate the predictor (dependent variable) using a Beta distribution
    # To skew towards 1, we set alpha > beta, controlled by the skew_factor.
    alpha = skew_factor
    beta = 1
    predictor = np.random.beta(alpha, beta, n_samples)
    
    # Combine features and predictor into a DataFrame
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_columns)
    df['predictor'] = predictor
    
    return df


# plt.figure(figsize=(8, 6))
# sns.histplot(df["predictor"], kde=True, bins=30, color='blue')
# plt.title('Distribution of Predictor (Skewness Check)', fontsize=14)
# plt.xlabel('Predictor', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.show()

# print((y_train, y_test))


# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class CustomFbetaMetric(tf.keras.metrics.Metric):
    def __init__(self, beta=1.0, threshold=0.5, name='custom_fbeta_score', **kwargs):
        super(CustomFbetaMetric, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.threshold = threshold
        
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
        # print(y_true)
        # y_pred_disc = tf.where(y_pred >= self.threshold, 1, 0)
        # y_true_disc = tf.where(y_true >= self.threshold, 1, 0)
        
        # y_pred_disc = tf.where(y_pred < self.threshold, 1, 0)
        # y_true_disc = tf.where(y_true < self.threshold, 1, 0)
        
        self.precision_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)
        self.recall_metric.update_state(y_true_disc, y_pred_disc, sample_weight=sample_weight)
        
        
        # self.precision_metric.update_state(y_true, y_pred_disc, sample_weight=sample_weight)
        # self.recall_metric.update_state(y_true, y_pred_disc, sample_weight=sample_weight)
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
        y_pred_disc = tf.cast(y_pred_disc, tf.float32)
        y_true_disc = tf.cast(y_true_disc, tf.float32)
        
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


# def fbeta_loss(y_true, y_pred): 
#     y_pred_disc = tf.where(y_pred >= self.threshold, 0, 1)
#     y_true_disc = tf.where(y_true >= self.threshold, 0, 1)
    
#     tp = tf.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32))
    
#     return(tp)
    
# fbeta_loss()




from sklearn.datasets import make_classification
import numpy as np

def generate_unbalanced_data(n_samples=100000, n_features=20, imbalance_ratio=0.1, random_state=None):
    """
    Generate an unbalanced binary classification dataset.

    Parameters:
    - n_samples: Total number of samples to generate.
    - n_features: Number of features for the dataset.
    - imbalance_ratio: Proportion of the minority class (between 0 and 0.5).
    - random_state: Random seed for reproducibility.

    Returns:
    - X: Feature matrix of shape (n_samples, n_features).
    - y: Labels array of shape (n_samples,), with a minority class proportion based on imbalance_ratio.
    """
    if imbalance_ratio <= 0 or imbalance_ratio >= 0.5:
        raise ValueError("Imbalance ratio must be between 0 and 0.5.")

    # Calculate the number of samples for each class
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    # Generate the dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),  # Adjusted number of informative features
        n_redundant=int(n_features * 0.2),    # Adjusted number of redundant features
        n_clusters_per_class=1,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        flip_y=0,  # No noise in the labels
        random_state=random_state
    )
    
    if np.sum(y) < n_samples * (1 - imbalance_ratio):
        # If the minority class is labeled as 0, flip the labels
        y = np.where(y == 0, 1, 0)
    
    return X, y

# Example usage:
# X, y = generate_unbalanced_data(n_samples=50000, n_features=20, imbalance_ratio=0.01, random_state=42)

# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

def fbeta_loss_function(threshold, beta=1):
    def fbeta_loss(y_true, y_pred): 
        
        y_pred_disc = tf.where(y_pred >= threshold, 0, 1)
        y_true_disc = tf.where(y_true >= threshold, 0, 1)
        
        tp = tf.reduce_sum(tf.cast(y_true_disc * y_pred_disc, dtype=tf.float32))
        tn = tf.reduce_sum(tf.cast((1-y_true_disc)*(1-y_pred_disc), dtype=tf.float32))
        fp = tf.reduce_sum(tf.cast((1-y_true_disc)*y_pred_disc, dtype=tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true_disc*(1-y_pred_disc), dtype=tf.float32))
        
        
        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())
        
        beta_squared = beta**2
        
        fb = ((1+beta_squared) * p * r)/((beta_squared * p) + r)
        
        return(1-fb)
    
    return(fbeta_loss)




def f1(y_true, y_pred):

    tp = tf.reduce_sum(tf.keras.backend.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = tf.reduce_sum(tf.keras.backend.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.keras.backend.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2*p*r / (p+r+tf.keras.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.keras.backend.mean(f1)

def weighted_mse_loss(y_true, y_pred):
    # Ensure y_true is not zero to avoid division by zero
    epsilon = tf.keras.backend.epsilon()  # A small constant to avoid division by zero
    weights = 1.0 / (y_true + epsilon)  # Weights inversely proportional to y_true

    # Calculate the squared error
    squared_error = tf.square(y_true - y_pred)
    
    # Calculate the weighted mean squared error
    weighted_mse = tf.reduce_mean(weights * squared_error)
    
    return weighted_mse


def quantile_loss_wrapper(q):
    def quantile_loss(y_true, y_pred):
        error = y_true - y_pred
        loss = tf.maximum(q * error, (q - 1) * error)
        return tf.reduce_mean(loss)
    return quantile_loss

def custom_weighted_mse(alpha):
    def loss_function(y_true, y_pred):
        # Calculate squared error
        squared_error = tf.square(y_true - y_pred)
        
        # Calculate weighting factor based on y_true
        weight = 1 + alpha * (1 - y_true)
        
        # Apply weighted MSE
        weighted_squared_error = weight * squared_error
        return tf.reduce_mean(weighted_squared_error)
    
    return loss_function

def custom_exponential_loss(alpha):
    def loss_function(y_true, y_pred):
        # Calculate the squared error (y_hat - y)^2
        squared_error = tf.abs(y_pred - y_true)
        
        # Calculate the exponential weighting term e^(-alpha * y)
        exponential_weight = tf.exp(-alpha * y_true)
        
        # Combine the exponential weight and the squared error
        loss = exponential_weight * squared_error
        
        return tf.reduce_mean(loss)
    
    return loss_function

def focal_loss_regres(y_true, y_pred):
    first = (1-y_true)**2
    second = (y_pred - y_true)**2
    return tf.reduce_mean(first*second)

def mae_poly_loss(alpha):
    def loss(y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        second = (1-y_true)**alpha
        
        return tf.reduce_mean(mae*second)
    return loss

df = generate_dataset(500000, 10)

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
X_train

model = tf.keras.Sequential([
    tf.keras.layers.Input((10,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

# print((X_train.shape, X_test.shape, y_train.shape, y_test.shape))       BinaryFocalCrossentropy weighted_mse_loss
model.compile(optimizer='adamax',
              loss=["mse"],
              metrics=[CustomFbetaMetric]
              )

history = model.fit(X_train, y_train, epochs=100)
history_df = pd.DataFrame(history.history)
# plt.plot(history_df["custom_fbeta_score"])
# plt.show()

# plt.plot(history_df["loss"])
# plt.show()
res = model.evaluate(X_test,  y_test, verbose=2)
y_pred = model.predict(X_test)
print(y_test)
residuals = y_test - y_pred

# plt.scatter(y_pred, residuals)
# plt.xlabel("Predicted Values")
# plt.ylabel("Residuals")
# plt.axhline(0, color='r', linestyle='--')
# plt.title("Residual Plot")
# plt.savefig("ResidualPlot.png")
# plt.show()
# print(mean_absolute_error(y_test, y_pred))

# plt.scatter(y_test, residuals)
# plt.xlabel("True Values")
# plt.ylabel("Residuals")
# plt.axhline(0, color='r', linestyle='--')
# plt.title("Residual Plot")
# plt.savefig("ResidualPlot_ytest.png")
# plt.show()

# sns.histplot(residuals, kde=True)
# plt.xlabel("Residuals")
# plt.title("Error Distribution")
# plt.savefig("ErrorDistribution.png")
# plt.show()

# sns.histplot(np.abs(residuals), kde=True)
# plt.xlabel("Residuals")
# plt.title("Error Distribution")
# plt.savefig("ErrorDistribution_abs.png")
# plt.show()
# plt.figure(figsize=(8, 6))
# sns.histplot(y_train, kde=True, bins=30, color='blue')
# plt.title('Distribution of Predictor (Skewness Check)', fontsize=14)
# plt.xlabel('Predictor', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.show()


print(predicted)




