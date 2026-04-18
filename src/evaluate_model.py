from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# 🔹 Confusion Matrix Plot
def plot_confusion_matrix(cm, labels, model_name):
    plt.figure()
    plt.imshow(cm, cmap='viridis')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()

    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Show numbers inside boxes
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="white")

    plt.show()


# 🔹 Feature Importance Plot (only for models like RandomForest)
def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        plt.figure()
        plt.barh(feature_names, importances)
        plt.xlabel("Importance")
        plt.title(f"{model_name} - Feature Importance")
        plt.show()


# 🔹 Main Evaluation Function
def evaluate(models, X_test, y_test, feature_names):
    labels = ['Declining', 'Improving', 'Stable']

    for name, model in models.items():
        predictions = model.predict(X_test)

        print(f"\nModel: {name}")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions, labels=labels)
        plot_confusion_matrix(cm, labels, name)

        # Feature Importance
        plot_feature_importance(model, feature_names, name)