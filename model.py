import os.path

import keras


def get_model():
    model = keras.Sequential([
        keras.layers.Conv2D(128, activation='relu', name='hidden', kernel_size=3, strides=1, padding='same', input_shape=(28, 28, 1)),
        # relu bc "everything unimportant is equally unimportant"
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(10, name='output', activation='softmax')  # needs 10 neurons bc this is a classification problem (classifying digits 0-9)
        # softmax on the output because need to convert the output to probabilities (which digit is most likely)
    ])
    if os.path.isfile('model.keras'):
        model.load_weights('model.keras')
    else:
        print("No model found, training from scratch")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    # adam faster to converge, sdg more accurate ALSO sparse bc categories aren't one-hot encoded
    return model
