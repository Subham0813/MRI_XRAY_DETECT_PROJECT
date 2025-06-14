import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, testX, testY, label_encoder, output_path=None):
    # Predict probabilities
    predictions = model.predict(testX)

    # Convert predictions and true labels to class indices
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(testY, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_
    Confusion_Matrix = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(Confusion_Matrix, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if output_path:
        plt.savefig(output_path)
    plt.show()

    # Print classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=5)
    print(report)

    return report
