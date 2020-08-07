import cv2
# image of a car
image_file = 'images2.jpg'
video_file = 'motorbike_dashcam.mp4'

# read image
img = cv2.imread(image_file)

# get video
video = cv2.VideoCapture(video_file)
# car  trained modeled xml file
# pretrained car classifier
carTracker = 'car_detector.xml'
# pedestrian trained modeled xml file
pedestrianTracker = 'pedesterian_detector.xml'
# cascade classifier
Car_Tracker = cv2.CascadeClassifier(carTracker)
pedestrian_tracker = cv2.CascadeClassifier(pedestrianTracker)

while True:

    (read_successfull, frame) = video.read()
    if read_successfull:
        frame_black_n_white = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        break

    # detect car
    car = Car_Tracker.detectMultiScale(frame_black_n_white)
    # detect pedeterian
    pedestrian = pedestrian_tracker.detectMultiScale(frame_black_n_white)

    # printing rectangles containing cars
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (0, 0, 225), 2)
    # printing rectangles containing pedestrian
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y),
                      (x + w, y + h), (225, 0, 0), 2)

    # show image for split second
    cv2.imshow("car image : ", frame)

    # wait until a key is pressed
    key = cv2.waitKey(1)
    if key == 13 or key == 27:
        break

# # show image for split second
# cv2.imshow("car image : ", img)

# # wait until a key is pressed
# cv2.waitKey()


# # convert image to black and white
# image_bnw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# # car classifier
# Car_Tracker = cv2.CascadeClassifier(classifier_file)

# # detect car
# car = Car_Tracker.detectMultiScale(image_bnw)
# print(car)

# # printing rectangles containing cars
# for (x, y, w, h) in car:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 225), 2)


# # show image for split second
# cv2.imshow("car image : ", img)

# # wait until a key is pressed
# cv2.waitKey()

print("code executed")
