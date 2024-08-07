To train such model you need to download images and anotations from this source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Or you can use your own source :-)

File gender_any_data.py can take a folder as argument and recognize all photos in that folder.

File gender_with_cv2.py is code for training the model with data from selebs folder with using cv2 for preprocessing.
File gender_eval.py is just a code for evalueting the model on some random data from selebs folder with cv2 preprocesing as well. 

File gender_nocv2.py is code for training the model with data from selebs folder with using tensorflow preprocessing functions.
File gender_eval1.py is just a code for evalueting the model on some random data from selebs folder with tensorflow preprocesing functions.

When you use trained model, it's better to use the same preprocess method which was used during training.

File add_training.py make additional training of the model
