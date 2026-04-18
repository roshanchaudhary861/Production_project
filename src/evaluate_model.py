from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate(models, X_test, y_test):
    accuracies = {}

    for name, model in models.items():
        predictions = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, predictions)
        accuracies[name] = acc

        print(f"\nModel: {name}")
        print("Accuracy:", acc)
        print(classification_report(y_test, predictions))

        # 📊 Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plot_confusion_matrix(cm, name)

    # 📊 Model Comparison Graph
    plot_model_comparison(accuracies)


# 🔹 Confusion Matrix Plot
def plot_confusion_matrix(cm, model_name):
    plt.figure()
    plt.imshow(cm)

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    labels = ["Declining", "Improving", "Stable"]
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    # Show numbers inside boxes
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j],
                     ha="center", va="center")

    plt.colorbar()
    plt.show()


# 🔹 Model Comparison Bar Chart
def plot_model_comparison(accuracies):
    names = list(accuracies.keys())
    scores = list(accuracies.values())

    plt.figure()
    plt.bar(names, scores)

    plt.title("Model Comparison (Accuracy)")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")

    plt.show()