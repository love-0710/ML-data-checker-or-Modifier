from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator settings
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rotation_range=20,  # Add rotation
    width_shift_range=0.2,  # Add width shift
    height_shift_range=0.2,  # Add height shift
    brightness_range=[0.8, 1.2]  # Adjust brightness
)


valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training Set
train_set = train_datagen.flow_from_directory(train_dir,
                                             target_size=(224, 224),  #change target size based on the model input size
                                             batch_size=10,
                                             class_mode='categorical')

# Validation Set
valid_set = valid_datagen.flow_from_directory(validation_dir,
                                              target_size=(224, 224),
                                              batch_size=10,
                                              class_mode='categorical',
                                              shuffle=False)

# Test Set
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(224, 224),
                                            batch_size=10,
                                            class_mode='categorical',
                                            shuffle=False)
