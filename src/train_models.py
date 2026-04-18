from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    models = {}

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    # SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    return models


def save_models(models):
    import pickle
    import os

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        with open(f"models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)