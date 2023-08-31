import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from model import get_model

data = tfds.load('mnist', split='train', as_supervised=True)  # as_supervised means it's a tuple and not a dict
test = tfds.load('mnist', split='test', as_supervised=True)
data = data.cache().shuffle(data.cardinality(), reshuffle_each_iteration=True).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
test = test.cache().shuffle(test.cardinality(), reshuffle_each_iteration=True).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
model = get_model()
checkpoint = keras.callbacks.ModelCheckpoint('model.keras', save_weights_only=True, save_best_only=True, mode='max', monitor='accuracy')
early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, mode='max')
model.fit(data, epochs=100, verbose=1, callbacks=[checkpoint, early_stop])
