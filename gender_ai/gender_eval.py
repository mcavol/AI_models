import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import random

IMG_WIDTH = 89
IMG_HEIGHT = 109
DATA_NUMBER = 20
TOTAL_DATA = 202599

def main():

    model = tf.keras.models.load_model('gender_model.h5')
    images = load_data(sys.argv[1], sys.argv[2])
    img_array = np.array(images)

    predictions = model.predict(img_array)

    for i in range(DATA_NUMBER):

        predicted_class = 'Male' if predictions[i] > 0.5 else 'Female'
        # Display the image with the predicted class
        plt.imshow(img_array[i])    #matplotlib can handle images after cv2, so no need for additional transformation
        #plt.imshow(images[i])    #can even plot images, not img_array
        plt.title(f'Predicted: {predicted_class}', fontsize=20)
        plt.axis('off')  # Hide axes
        #plt.show()

        # Save the plot to a file
        plt.savefig(f'prediction_{i}.png')
        plt.close()




def load_data(data_dir, attr_list):
    images = []
    
    #open txt file with list of images and their attributes
    with open(attr_list) as f:
        lines = f.readlines()

    #get list of DATA_NUMBER random numbers from 1 to TOTAL_DATA number which we have (so 10000 numbers in range(1, 202599))
    image_numbers = get_random_list()

    #take a number from numbers list and it will be index of the random line in txt with images and attributes
    for number in image_numbers:
        splited_line = lines[number].split()

        #get image name from the line of txt file and use image with this name, read image, resize it and add to images list
        if os.path.isdir(data_dir):
            filename = splited_line[0]
            image_path = os.path.join(data_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)

    return images



def get_random_list():
    image_numbers = random.sample(range(1, TOTAL_DATA+1), DATA_NUMBER)
    return image_numbers

if __name__ == "__main__":
    main()