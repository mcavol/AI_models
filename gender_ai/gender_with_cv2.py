import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 89
IMG_HEIGHT = 109
NUM_CATEGORIES = 1
TEST_SIZE = 0.3
DATA_NUMBER = 10000
TOTAL_DATA = 202599


def main():

    # Check command-line arguments
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1], sys.argv[2])

    # Split data into training and testing sets

    #labels = tf.keras.utils.to_categorical(labels)     #to_categorial splits labels to many categories for softmax function and onehot data

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE    #np.array(images) take bunch of images and make array from them
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 4:
        filename = sys.argv[3]
        model.save(filename)
        print(f"Model saved to {filename}.")

    # Debugging: Print predictions and true labels
    predictions = model.predict(x_test)
    for i in range(10):
        print(f"Prediction: {predictions[i]}, True Label: {y_test[i]}")
         # Display the image with the predicted class
        plt.imshow(x_test[i])    #matplotlib can handle images after cv2, so no need for additional transformation
        plt.title(f'Predicted: {predictions[i]}', fontsize=20)
        plt.axis('off')  # Hide axes
        #plt.show()

        # Save the plot to a file
        plt.savefig(f'prediction_{i}.png')
        plt.close()


def load_data(data_dir, attr_list):
    
    images = []
    labels = []
    
    #open txt file with list of images and their attributes
    with open(attr_list) as f:
        lines = f.readlines()

    #find index of "male" collumn there
    male_attr = lines[0].split().index("Male") + 1  #add +1 because in next lines there will be +1 collumn

    #get list of DATA_NUMBER random numbers from 1 to TOTAL_DATA number which we have (so 10000 numbers in range(1, 202599))
    image_numbers = get_random_list()

    #take a number from numbers list and it will be index of the random line in txt with images and attributes
    for number in image_numbers:
        splited_line = lines[number].split()

        #get male attribute from the line of txt file and add it to labels
        if splited_line[male_attr] == "1":
            labels.append(1)
        else:
            labels.append(0)

        #get image name from the line of txt file and use image with this name, read image, resize it and add to images list
        if os.path.isdir(data_dir):
            filename = splited_line[0]
            image_path = os.path.join(data_dir, filename)
            image = cv2.imread(image_path)    # loads the single image as a NumPy array in BGR format
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image/255.0    #normalize image
            images.append(image)
                
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            64, (2, 2), activation="relu"
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 


        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Add an output layer with output units for all 2 genders
        #tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")   #softmax function uses when we have many categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid")   #sigmoid function uses when we have binary data, only 0 and 1 as categories
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        #loss="categorical_crossentropy",   #categorial_crossentropy use for categorial data
        loss ="binary_crossentropy",      #binary_crossentropy use forbinary data
        metrics=["accuracy"]
    )

    return model


def get_random_list():
    lines = random.sample(range(1, TOTAL_DATA+1), DATA_NUMBER)
    return lines

if __name__ == "__main__":
    main()