import tensorflow as tf
import numpy as np
import cv2
import logging

class Predictor:
    def __init__(self, Validation_Model, Disease_Predictor):

        self.Validation_Model = Validation_Model

        self.Disease_Predictor = Disease_Predictor

        logging.basicConfig(filename="Logger.log",
                        format='%(asctime)s %(message)s', filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    def load_classes(self, predicted_class_index):

        Classes = {0 : 'bacterial_leaf_blight', 1 : 'bacterial_leaf_streak', 2 : 'bacterial_panicle_blight', 3 : 'blast', 4 : 'brown_spot', 5 : 'dead_heart', 6 : 'downy_mildew', 7 : 'hispa', 8 : 'normal', 9 : 'tungro'}
        
        return Classes[predicted_class_index]

    def Disease_predictor_Preprocessor(self, image): 

        # Perform necessary Preprocessing steps on the image to predict the disease

        img_dim = 128

        resized = cv2.resize(image, (img_dim, img_dim))

        normalized = resized / 255.0

        return np.array([normalized])
    
    def Validation_Preprocessor(self, image):

        # Perform necessary Preprocessing steps on the image to find the image is paddy or not

        img = tf.keras.preprocessing.image.img_to_array(image)

        img = np.expand_dims(img, axis=0)

        images = np.vstack([img])

        return images
    
    def Predict(self, image_path):

        logging.info("Prediction Started ....")

        try:
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
            logging.info("Image Loaded Successfully")
        except:
            logging.error("Problem While Loading the Image")

        preprocessed_image_1 = self.Validation_Preprocessor(image)

        prediction_1 = self.Validation_Model.predict(preprocessed_image_1, batch_size=10)

        logging.info(f"Validation Prediction --> {prediction_1}")


        if prediction_1[0]>0.5 :

            preprocessed_image_2 = self.Validation_Preprocessor(image)

            prediction_2 = self.Disease_Predictor.predict(preprocessed_image_2)

            predicted_class_index = np.argmax(prediction_2)

            print(predicted_class_index)
            
            predicted_class = self.load_classes(predicted_class_index)

            logging.info("Prediction Completed ....")

            return {"isPaddy" : True, "classPrediced" : predicted_class}
        
        else :

            logging.info("Given image is not related to paddy plant ....")

            return {"isPaddy" : False, "classPrediced" : None}