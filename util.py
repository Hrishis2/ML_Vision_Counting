import cv2
import mediapipe as mp
import time
import tensorflow as tf




def edge_detection(img_path):
    # Read the original image
    img = cv2.imread(img_path)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    
    return (sobelx, sobely, sobelxy, edges)




def predict(model, crop_img):
    rgb_tensor = tf.convert_to_tensor(crop_img, dtype=tf.float32)
    # rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    print(rgb_tensor.shape)
    prediction = model.predict(crop_img)
    return prediction




def bound_coordinate(coord, dim, w, h):
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




def hand_silhouetting(img: list, args, img_path: str=None) -> list:

    if img_path is not None:
        success, img_read = cv2.imread(img_path)
        if not success:
            raise Exception("Failed to read image path for hand sillhouetting.")
        else:
            img = img_read

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    crop_minx, crop_maxx, crop_miny, crop_maxy = 0, 1, 0, 1


    if results.multi_hand_landmarks:

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
        if args.show_silhouetting:
            cv2.rectangle(img, (minimum_x, minimum_y), (maximum_x, maximum_y), (255,0,0), 10)
        crop_minx = minimum_x
        crop_maxx = maximum_x
        crop_miny = minimum_y
        crop_maxy = maximum_y


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    h, w, c = img.shape
    crop_minx = bound_coordinate(crop_minx, 0, w, h)
    crop_maxx = bound_coordinate(crop_maxx, 0, w, h)
    crop_miny = bound_coordinate(crop_miny, 1, w, h)
    crop_maxy = bound_coordinate(crop_maxy, 1, w, h)

    threshold = 20 # 50 pixels for cropping threshold
    crop_size = max(crop_maxy-crop_miny, crop_maxx-crop_minx)
    crop_miny_threshold = bound_coordinate(crop_miny-threshold, 1, w, h)
    crop_minx_threshold = bound_coordinate(crop_minx-threshold, 0, w, h)
    crop_maxy_threshold = bound_coordinate(crop_miny+crop_size+threshold, 1, w, h)
    crop_maxx_threshold = bound_coordinate(crop_minx+crop_size+threshold, 0, w, h)
    crop_img = img[crop_miny_threshold:crop_maxy_threshold, crop_minx_threshold:crop_maxx_threshold]

    # crop_img = cv2.resize(img, (64, 64), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    if args.debug:
        cv2.imshow("sillhouetted", crop_img)
    
    return crop_img