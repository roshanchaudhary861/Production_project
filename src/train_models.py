from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=2000)
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt


# 🔹 Split + scale data
def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# 🔹 Train models
def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


# 🔹 Save models
def save_models(models):
    for name, model in models.items():
        with open(f"models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)


# 🔹 Cross-validation
def cross_validate_models(models, X, y):
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        print(f"\n{name} Cross-Validation Accuracy:")
        print("Scores:", scores)
        print("Mean:", scores.mean())

        results[name] = scores

    return results


# 🔹 Plot Cross-validation graph
def plot_cv_results(results):
    plt.figure()

    for name, scores in results.items():
        plt.plot(scores, marker='o', label=name)

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Results")
    plt.legend()
    plt.show()