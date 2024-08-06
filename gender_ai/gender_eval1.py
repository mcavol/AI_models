import tensorflow as tf
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
    image_numbers = get_random_list()
    with open(sys.argv[2]) as f:
        lines = f.readlines()
    
    for i, number in enumerate(image_numbers):
        splited_line = lines[number].split()
        if os.path.isdir(sys.argv[1]):
            filename = splited_line[0]
            img_path = os.path.join(sys.argv[1], filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array/255.0
            # Add batch dimension because before it was added by np.array, but for single image it doesn't add batch dim, only for multiple images
            img_array = tf.expand_dims(img_array, 0)
            predictions = model.predict(img_array)

            # Get the predicted class (assuming binary classification)
            predicted_class = 'Male' if predictions[0][0] > 0.5 else 'Female'
            print (predictions)

            # Display the image with the predicted class
            
            # can plot just images before preprocessing
            #plt.imshow(img)

            #or can take image_array which is tensor, make it NumPy array and then make all colors [0,255] again and make it in unit8 format
            display_img = (img_array[0].numpy() * 255).astype('uint8')    #matplotlib can't handle images after tensorflow preprocessing, so need an additional transformation
            plt.imshow(display_img)
            plt.title(f'Predicted: {predicted_class}', fontsize=20)
            plt.axis('off')  # Hide axes
            #plt.show()

            # Save the plot to a file
            plt.savefig(f'prediction_{i}.png')
            plt.close()


def get_random_list():
    image_numbers = random.sample(range(1, TOTAL_DATA+1), DATA_NUMBER)
    return image_numbers

if __name__ == "__main__":
    main()