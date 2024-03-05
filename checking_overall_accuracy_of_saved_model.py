import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths to your train, validation, and test data directories
train_data_dir = 'ASD_v4_datasets/train'
val_data_dir = 'ASD_v4_datasets/valid'
test_data_dir = 'ASD_v4_datasets/test'

# Create an ImageDataGenerator for train, validation, and test data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the Inception model
model_path = 'resized_inception.h5'

# Load train, validation, and test images and their labels using the ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Load the model
model = tf.keras.models.load_model(model_path)


# Print model summary
print("Model Summary:")
model.summary()

# Get predictions for train, validation, and test data
train_preds = model.predict(train_generator)
val_preds = model.predict(val_generator)
test_preds = model.predict(test_generator)

# Convert predictions to binary labels
train_pred_labels = (train_preds > 0.5).astype(int)
val_pred_labels = (val_preds > 0.5).astype(int)
test_pred_labels = (test_preds > 0.5).astype(int)

# True labels
train_true_labels = train_generator.classes
val_true_labels = val_generator.classes
test_true_labels = test_generator.classes

# Calculate confusion matrices
train_conf_matrix = confusion_matrix(train_true_labels, train_pred_labels)
val_conf_matrix = confusion_matrix(val_true_labels, val_pred_labels)
test_conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)

# Plot confusion matrices
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(train_conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Train Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.subplot(1, 3, 2)
sns.heatmap(val_conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.subplot(1, 3, 3)
sns.heatmap(test_conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.tight_layout()
plt.show()


# Calculate precision, recall, and F1-score
precision = precision_score(test_true_labels, test_pred_labels)
recall = recall_score(test_true_labels, test_pred_labels)
f1 = f1_score(test_true_labels, test_pred_labels)

# Print evaluation metrics
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
