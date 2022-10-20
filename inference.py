import tensorflow as tf
import cv2
import util


if __name__ == "__main__":
    # initialize the camera
    cam = cv2.VideoCapture(0)   # 0 -> index of camera

    model = tf.keras.models.load_model('saved_model/model')
    
    while True:
        success, img = cam.read()
        if not success:
          raise Exception("CV2 failed to read the video.")

        img = util.hand_silhouetting(img)

        cv2.imshow('frame', img)

        # rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
        # print(rgb_tensor.shape)

        # prediction = model.predict(img)
        # print(prediction)

        # for row in range(len(img)):
        #     for col in range(len(img)):
        #         pixel = img[row][col]
        #         sum_p = sum(pixel)
        #         avg_p = pixel[0]*pixel[1]*pixel[2]*(1/3)
        #         img[row][col] = [avg_p/np.linalg.norm(sum_p)]
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break