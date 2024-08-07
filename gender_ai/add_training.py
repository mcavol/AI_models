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

    #labels = tf.keras.utils.to_categorical(labels)    #to_categorial splits labels to many categories for softmax function and onehot data

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE    #np.array(images) take bunch of images and make array from them
    )

    # Get a compiled neural network
    model = tf.keras.models.load_model('gender_model.h5')

    model.compile(
        optimizer="adam",
        #loss="categorical_crossentropy",    #categorial_crossentropy use for categorial data
        loss ="binary_crossentropy",    #binary_crossentropy use forbinary data
        metrics=["accuracy"])

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
        display_img = (x_test[i] * 255).astype('uint8')    #matplotlib can't handle images after tensorflow preprocessing, so need an additional transformation
        # Display the image with the predicted class
        plt.imshow(display_img)
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
            img_path = os.path.join(data_dir, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img)    #take the single image and make NumPy array from it
            img_array = img_array/255.0
            images.append(img_array)
                
    return images, labels



def get_random_list():
    lines = random.sample(range(1, TOTAL_DATA+1), DATA_NUMBER)
    return lines

if __name__ == "__main__":
    main()