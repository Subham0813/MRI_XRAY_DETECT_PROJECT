from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

def build_cnn_model(image_size=150):
    model = Sequential()
    
    # Convolutional + Pooling layers
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu", 
                     kernel_initializer='he_normal', input_shape=(image_size, image_size, 3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Fully connected layers (ANN part)
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation="softmax"))  # 3 output classes

    # Compile the model
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



def build_mri_model(num_classes):
    IMAGE_SIZE = 128

    base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last 3 layers
    base_model.layers[-2].trainable = True
    base_model.layers[-3].trainable = True
    base_model.layers[-4].trainable = True

    model = Sequential()
    model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model
