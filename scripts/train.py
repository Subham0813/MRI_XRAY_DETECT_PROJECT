import matplotlib.pyplot as plt

def train_model(model, trainX, trainY, valX, valY, epochs=30, batch_size=40):
    history = model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def plot_training_history(history, output_path=None):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()
