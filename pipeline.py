import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


# Binary classification model
class BinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Multiclass classification model
class MulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = RandomForestClassifier()  # Using RandomForestClassifier instead

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# Custom transformer to extract probabilities from binary classifier
class BinaryPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_classifier = BinaryClassifier()
        self.probability_extractor = ProbabilityExtractor(self.binary_classifier)

    def fit(self, X, y=None):
        self.binary_classifier.fit(X, y)
        self.probability_extractor.fit(X, y)
        return self

    def transform(self, X):
        probabilities = self.probability_extractor.transform(X)
        return probabilities


# Custom transformer to extract probabilities from binary classifier
class ProbabilityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1].reshape(-1, 1)  # Extracting probabilities for class 1


# Custom transformer for multiclass classification
class MulticlassPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.multiclass_classifier = MulticlassClassifier()

    def fit(self, X, y=None):
        self.multiclass_classifier.fit(X, y)
        return self

    def transform(self, X):
        predictions = self.multiclass_classifier.predict(X)
        return predictions.reshape(-1, 1)


df = pd.read_csv("Cleaning_Data.csv")

from sklearn.preprocessing import LabelEncoder

# Initializing LabelEncoder
label_encoder = LabelEncoder()

# Encoding column 'status'
df['status'] = label_encoder.fit_transform(df['status'])

#0:acquired,1:closed,2:IPO,3:operating
X = df.drop(["isClosed", "status"], axis=1)
y_binary = df["isClosed"]
y_multiclass = df["status"]

# Splitting data into train and test sets
X_train, X_test, y_binary_train, y_binary_test, y_multiclass_train, y_multiclass_test = train_test_split(
    X, y_binary, y_multiclass, test_size=0.2, random_state=42
)
closed = y_binary.value_counts();
print(closed)
status = y_multiclass.value_counts();
print(status)
# Defining pipeline
binary_pipeline = BinaryPipeline()
multiclass_pipeline = MulticlassPipeline()

# Combining pipelines
combined_pipeline = ColumnTransformer(
    transformers=[
        ("binary_pipeline", binary_pipeline, slice(0, len(X.columns))),  # Step 1: Binary pipeline
        ("multiclass_pipeline", multiclass_pipeline, slice(0, len(X.columns)))  # Step 2: Multiclass pipeline
    ]
)

# Final estimator
final_estimator = MulticlassClassifier()

# Combining ColumnTransformer and final estimator
full_pipeline = Pipeline(
    steps=[
        ("feature_engineering", combined_pipeline),
        ("final_estimator", final_estimator)
    ]
)

# Training the full pipeline
full_pipeline.fit(X_train, y_multiclass_train)

# Predictions
y_pred = full_pipeline.predict(X_test)

# Evaluating the model
print(classification_report(y_multiclass_test, y_pred))

import joblib
from joblib import dump, load

# Saving the pipeline
dump(full_pipeline, 'full_pipeline.joblib')

# Later we can load it
loaded_pipeline = load('full_pipeline.joblib')

with open('full_pipeline.joblib', 'rb') as f:
    obj = joblib.load(f)

# For prediction
y_pred = loaded_pipeline.predict(X_test)
print(y_pred)
