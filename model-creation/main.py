import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_and_preprocess_image(image_path, image_width, image_height):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)  # Automatically detects image format
        img = tf.image.resize(img, [image_width, image_height])
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error loading image: {image_path}. Exception: {e}")
        # Return a placeholder black image when an error occurs
        img_shape = (image_width, image_height, 3)
        return tf.zeros(img_shape, dtype=tf.float32)

def load_set(folder_path, image_width, image_height, batch_size):
    image_paths = []
    labels = []

    diseases = os.listdir(folder_path)

    for index, disease in enumerate(diseases):
        disease_folder = os.path.join(folder_path, disease)
        image_files = os.listdir(disease_folder)
        image_paths.extend([os.path.join(disease_folder, file) for file in image_files if file.endswith('.jpg') or file.endswith('.JPG')])
        labels.extend([index] * len(image_files))

        # Convert images to JPEG format
        for image_file in image_files:
            image_path = os.path.join(disease_folder, image_file)
            if not image_file.lower().endswith('.jpg'):
                img = Image.open(image_path)
                jpeg_path = os.path.splitext(image_path)[0] + '.jpg'
                img = img.convert('RGB')
                img.save(jpeg_path, format='JPEG')
                os.remove(image_path)

    image_paths = tf.constant(image_paths)
    labels = tf.one_hot(labels, depth=11)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size=len(image_paths))

    def load_and_preprocess_image_wrapper(image_path, label):
        img = tf.py_function(load_and_preprocess_image, [image_path, image_width, image_height], tf.float32)
        return img, label

    dataset = dataset.map(load_and_preprocess_image_wrapper)
    dataset = dataset.filter(lambda img, label: img is not None)  # Remove images with errors
    dataset = dataset.batch(batch_size)

    return dataset

def create_model(image_width, image_height):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  
    model.add(layers.Dense(11, activation='softmax'))

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


def main():

    # Specify image and model parameters
    image_width, image_height = 256, 256
    batch_size = 32
    epochs = 20

    # Specify folder location and load dataset
    train_folder = ""
    validation_folder = ""
    training_set = load_set(train_folder, image_width, image_height, batch_size)
    validation_set = load_set(validation_folder, image_width, image_height, batch_size)

    # Create and fit model
    model = create_model(image_width, image_height)

    model.fit(training_set, epochs=epochs)

    model.save("")

    # Evaluate the model
    evaluation_results = model.evaluate(validation_set)
    validation_loss = evaluation_results[0]
    validation_accuracy = evaluation_results[1]

    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")

if __name__ == '__main__':
    main()
