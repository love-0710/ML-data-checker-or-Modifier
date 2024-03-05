import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('inception.h5')

# Define the new input shape
new_input_shape = (224, 224, 3)  # Assuming 3 channels for RGB images #change the size as per your requirement

# Create a new input layer with the desired input shape
new_input_layer = tf.keras.layers.Input(shape=new_input_shape)

# Resize the input to match the original input size (150x150)
resized_input = tf.keras.layers.experimental.preprocessing.Resizing(150, 150)(new_input_layer)

# Feed the resized input to the original Inception model
outputs = model(resized_input)

# Create a new model that takes the new input layer and outputs the same output as the original model
new_model = tf.keras.models.Model(inputs=new_input_layer, outputs=outputs)


# Print the summary of the updated model
print("Summary of the resized inception model:")
new_model.summary()

# Save the new model with the updated input size
new_model.save('resized_inception.h5')

# Print completion message
print("New model with updated input size has been saved as 'resized_inception.h5'")
