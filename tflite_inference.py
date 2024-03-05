import tensorflow as tf
import numpy as np

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare sample input data (replace with your actual input data)
sample_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], sample_input)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the output
print("Output:", output_data)
