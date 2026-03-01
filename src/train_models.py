from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def prepare_data(df):

    features = [
        'TotalSteps',
        'TotalDistance',
        'VeryActiveMinutes',
        'FairlyActiveMinutes',
        'LightlyActiveMinutes',
        'SedentaryMinutes',
        'Calories'
    ]

    X = df[features]
    y = df['HabitLabel']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_models(X_train, y_train):

    models = {}

    # Random Forest (no scaling needed)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # Logistic Regression (with scaling)
    log_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000))
    ])
    log_model.fit(X_train, y_train)
    models['LogisticRegression'] = log_model

    # SVM (with scaling)
    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC())
    ])
    svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model

    return models


def save_models(models):
    for name, model in models.items():
        joblib.dump(model, f"models/{name}.pkl")