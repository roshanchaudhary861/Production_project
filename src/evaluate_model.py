from sklearn.metrics import accuracy_score, classification_report

def evaluate(models, X_test, y_test):

    for name, model in models.items():
        predictions = model.predict(X_test)

        print(f"\nModel: {name}")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))