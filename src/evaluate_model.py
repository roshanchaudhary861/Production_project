from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate(models, X_test, y_test):

    for name, model in models.items():

        # Predictions
        predictions = model.predict(X_test)

        # Print results
        print(f"\nModel: {name}")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))

        # 🔥 Confusion Matrix
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            cmap="Blues"
        )

        plt.title(f"{name} - Confusion Matrix")
        plt.show()