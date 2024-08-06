import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random

IMG_WIDTH = 89
IMG_HEIGHT = 109
def main():
    model = tf.keras.models.load_model('gender_model.h5')
    folder = input("Folder name: ")
    images = np.array(get_images(folder))
    predictions = model.predict(images)

    #or can take image_array which is tensor, make it NumPy array and then make all colors [0,255] again and make it in unit8 format
    for i, image in enumerate(images):
        display_img = (image * 255).astype('uint8')    #matplotlib can't handle images after tensorflow preprocessing, so need an additional transformation

        # Get the predicted class (assuming binary classification)
        predicted_class = 'Male' if predictions[i] > 0.5 else 'Female'

        # Display the image with the predicted class
        plt.imshow(display_img)
        plt.title(f'Predicted: {predicted_class}', fontsize=20)
        plt.axis('off')  # Hide axes
        #plt.show()

        # Save the plot to a file
        plt.savefig(f'prediction_{i}.png')
        plt.close()
    




def get_images(folder):
    if os.path.isdir(folder):
        image_names = os.listdir(folder)
        images = []
    
        for i, name in enumerate(image_names):
            
            img_path = os.path.join(folder, name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array/255.0
            # Add batch dimension because before it was added by np.array, but for single image it doesn't add batch dim, only for multiple images
            #img_array = tf.expand_dims(img_array, 0)
            images.append(img_array)
        return images

            

if __name__ == "__main__":
    main()