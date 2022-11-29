from configparser import Interpolation
import tensorflow as tf
import cv2
import util
import argparse
import numpy as np


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser("Inference for NN")
    argument_parser.add_argument("-d", "--debug", default=False, action="store_true", help="Boolean to enable verbosity--i.e. debugging")
    argument_parser.add_argument("-s", "--show_silhouetting", default=False, action="store_true", help="Boolean to enable showing silhouetting")
    args = argument_parser.parse_args()



    # initialize the camera
    cam = cv2.VideoCapture(0)   # 0 -> index of camera

    model = tf.keras.models.load_model('saved_model/model')
    
    while True:
        success, img = cam.read()
        if not success:
          raise Exception("CV2 failed to read the video.")

        image = util.hand_silhouetting(img, args)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, [256,256])
        img = tf.cast(img, tf.float32) / 255.
        img = np.expand_dims(img, axis=0)

        cv2.imshow('frame', image)

        # print(img.shape)
        predictions = model.predict(img).tolist()[0]
        max_confidence = max(predictions)
        prediction = predictions.index(max_confidence)
        print("Prediction: " + str(prediction) + " fingers -> " + str(max_confidence*100) + "%" + " confidence\n")
        # print(max)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break