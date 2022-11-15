import cv2
import mediapipe as mp
import time
import math
import tensorflow as tf




cap = cv2.VideoCapture(0)

model = tf.keras.models.load_model('saved_model/model')

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0



def bound(coord, dim, w, h):
    if coord < 0:
        coord = 0
        return 0
    
    if not dim and coord > w:
        coord = w
        return w
    if dim and coord > h:
        coord = h
        return h

    return coord


def predict(model, crop_img):
    rgb_tensor = tf.convert_to_tensor(crop_img, dtype=tf.float32)
    # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    print(rgb_tensor.shape)
    prediction = model.predict(crop_img)
    return prediction


while True:
    success, img = cap.read()
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = img

    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    


    if results.multi_hand_landmarks:
        # for handLms in results.multi_hand_landmarks:
        #     for id, lm in enumerate(handLms.landmark):
        #         #print(id,lm)
        #         h, w, c = img.shape
        #         cx, cy = int(lm.x *w), int(lm.y*h)
        #         #if id ==0:
        #         cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

        #         mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # find bounding box
        h, w, c = img.shape
        minimum_x = w
        maximum_x = 0
        minimum_y = h
        maximum_y = 0
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                minimum_x = min(x, minimum_x)
                maximum_x = max(x, maximum_x)
                minimum_y = min(y, minimum_y)
                maximum_y = max(y, maximum_y)

        minimum_x = int(minimum_x)
        maximum_x = int(maximum_x)
        minimum_y = int(minimum_y)
        maximum_y = int(maximum_y)

        # draw rectangle
        # cv2.rectangle(img, (minimum_x, minimum_y), (maximum_x, maximum_y), (255,0,0), 10)
        crop_minx = minimum_x
        crop_maxx = maximum_x
        crop_miny = minimum_y
        crop_maxy = maximum_y

        


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.imshow("Image", img)

    h, w, c = img.shape
    crop_minx = bound(crop_minx, 0, w, h)
    crop_maxx = bound(crop_maxx, 0, w, h)
    crop_miny = bound(crop_miny, 1, w, h)
    crop_maxy = bound(crop_maxy, 1, w, h)

    threshold = 20 # 50 pixels for cropping threshold
    crop_size = max(crop_maxy-crop_miny, crop_maxx-crop_minx)
    crop_miny_threshold = bound(crop_miny-threshold, 1, w, h)
    crop_minx_threshold = bound(crop_minx-threshold, 0, w, h)
    crop_maxy_threshold = bound(crop_miny+crop_size+threshold, 1, w, h)
    crop_maxx_threshold = bound(crop_minx+crop_size+threshold, 0, w, h)
    crop_img = img[crop_miny_threshold:crop_maxy_threshold, crop_minx_threshold:crop_maxx_threshold]

    crop_img = cv2.resize(img, (64, 64), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    imS = cv2.resize(crop_img, (960, 540))
    cv2.imshow("cropped", imS)

    # prediction = predict(model, crop_img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
