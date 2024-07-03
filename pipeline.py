import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# Load the data
df = pd.read_csv("Cleaning_Data.csv")

# Separate features and target
X = df.drop(["status", "isClosed"], axis=1)
y = df["status"]

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['bool']).columns

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models to try
models = {
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)

}


# Function to create the full pipeline
def create_pipeline(model):
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('classifier', model)
    ])


# Hyperparameter grids for each model
param_grids = {

    'XGBoost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.3]
    },

    'LightGBM': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [-1, 5, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.3]
    },

    'RandomForest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    pipeline = create_pipeline(model)
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
    results[model_name] = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'report': report
    }
    print(f"{model_name} - Best parameters: {grid_search.best_params_}")
    print(f"{model_name} - Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n")

# Find the best performing model
best_model_name = max(results, key=lambda x: results[x]['report']['weighted avg']['f1-score'])
best_model = results[best_model_name]['model']

print(f"Best performing model: {best_model_name}")

# Save the best model
joblib.dump(best_model, 'best_startup_status_model.joblib')

# Feature importance for the best model
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    feature_importance = best_model.named_steps['classifier'].feature_importances_
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

    # Ensure feature_importance and feature_names have the same length
    min_length = min(len(feature_importance), len(feature_names))
    feature_importance = feature_importance[:min_length]
    feature_names = feature_names[:min_length]

    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    print("Top 10 most important features:")
    print(feature_importance_df.head(10))
else:
    print("Feature importance is not available for this model.")







# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # Binary classification model
# class BinaryClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self):
#         self.model = RandomForestClassifier()
#
#     def fit(self, X, y):
#         self.model.fit(X, y)
#         return self
#
#     def predict(self, X):
#         return self.model.predict(X)
#
#     def predict_proba(self, X):
#         return self.model.predict_proba(X)
#
#
# # Multiclass classification model
# class MulticlassClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self):
#         self.model = RandomForestClassifier()  # Using RandomForestClassifier instead
#
#     def fit(self, X, y):
#         self.model.fit(X, y)
#         return self
#
#     def predict(self, X):
#         return self.model.predict(X)
#
#
# # Custom transformer to extract probabilities from binary classifier
# class BinaryPipeline(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.binary_classifier = BinaryClassifier()
#         self.probability_extractor = ProbabilityExtractor(self.binary_classifier)
#
#     def fit(self, X, y=None):
#         self.binary_classifier.fit(X, y)
#         self.probability_extractor.fit(X, y)
#         return self
#
#     def transform(self, X):
#         probabilities = self.probability_extractor.transform(X)
#         return probabilities
#
#
# # Custom transformer to extract probabilities from binary classifier
# class ProbabilityExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self, model):
#         self.model = model
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         probabilities = self.model.predict_proba(X)
#         return probabilities[:, 1].reshape(-1, 1)  # Extracting probabilities for class 1
#
#
# # Custom transformer for multiclass classification
# class MulticlassPipeline(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.multiclass_classifier = MulticlassClassifier()
#
#     def fit(self, X, y=None):
#         self.multiclass_classifier.fit(X, y)
#         return self
#
#     def transform(self, X):
#         predictions = self.multiclass_classifier.predict(X)
#         return predictions.reshape(-1, 1)
#
#
# df = pd.read_csv("Cleaning_Data.csv")
#
# from sklearn.preprocessing import LabelEncoder
#
# # Initializing LabelEncoder
# label_encoder = LabelEncoder()
#
# # Encoding column 'status'
# df['status'] = label_encoder.fit_transform(df['status'])
#
# #0:acquired,1:closed,2:IPO,3:operating
# X = df.drop(["isClosed", "status"], axis=1)
# y_binary = df["isClosed"]
# y_multiclass = df["status"]
#
# # Splitting data into train and test sets
# X_train, X_test, y_binary_train, y_binary_test, y_multiclass_train, y_multiclass_test = train_test_split(
#     X, y_binary, y_multiclass, test_size=0.2, random_state=42
# )
# closed = y_binary.value_counts();
# print(closed)
# status = y_multiclass.value_counts();
# print(status)
# # Defining pipeline
# binary_pipeline = BinaryPipeline()
# multiclass_pipeline = MulticlassPipeline()
#
# # Combining pipelines
# combined_pipeline = ColumnTransformer(
#     transformers=[
#         ("binary_pipeline", binary_pipeline, slice(0, len(X.columns))),  # Step 1: Binary pipeline
#         ("multiclass_pipeline", multiclass_pipeline, slice(0, len(X.columns)))  # Step 2: Multiclass pipeline
#     ]
# )
#
# # Final estimator
# final_estimator = MulticlassClassifier()
#
# # Combining ColumnTransformer and final estimator
# full_pipeline = Pipeline(
#     steps=[
#         ("feature_engineering", combined_pipeline),
#         ("final_estimator", final_estimator)
#     ]
# )
#
# # Training the full pipeline
# full_pipeline.fit(X_train, y_multiclass_train)
#
# # Predictions
# y_pred = full_pipeline.predict(X_test)
#
# # Evaluating the model
# print(classification_report(y_multiclass_test, y_pred))
#
# import joblib
# from joblib import dump, load
#
# # Saving the pipeline
# dump(full_pipeline, 'full_pipeline.joblib')
#
# # Later we can load it
# loaded_pipeline = load('full_pipeline.joblib')
#
# with open('full_pipeline.joblib', 'rb') as f:
#     obj = joblib.load(f)
#
# # For prediction
# y_pred = loaded_pipeline.predict(X_test)
# print(y_pred)

