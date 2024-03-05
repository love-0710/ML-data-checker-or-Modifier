import tensorflow as tf

# Load the Keras model from the HDF5 file
keras_model = tf.keras.models.load_model('VGG16.h5')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
