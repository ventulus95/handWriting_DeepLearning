import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model


img = cv2.imread("11.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_checker = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 175, 45)
# img_checker = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 25)
plt.imshow(img_checker)
plt.show()
_, contours, hierarchy = cv2.findContours(img_checker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(contour) for contour in contours]

for rect in rects:
    if rect[2]*rect[3] < 1000:
        continue
    print(rect)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    if rect[2] < 50:
        margin = 100
    else:
        margin = 60
    roi = img_checker[rect[1]-margin:rect[1]+rect[3]+margin, rect[0]-margin:rect[0]+rect[2]+margin]
    # cv2.imshow("Resulting Image with Rectangular ROIs", img)
    # cv2.waitKey()
    # Resize the image
    try:
        roi = cv2.resize(roi, (28, 28),  cv2.INTER_AREA)
    except Exception as e:
        print(str(e))
    model = load_model('model.h5')
    roi = roi/255.0
    img_input = roi.reshape(1, 28, 28, 1)
    prediction = model.predict(img_input)
    num = np.argmax(prediction)
    print(num)
    location = (rect[0]+rect[2], rect[1] + 20)
    cv2.putText(img, str(num), location, cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)

cv2.imshow("Resulting Image with Rectangular ROIs", img)
cv2.waitKey()

# img_binary = cv2.adaptiveThreshold(img_morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)



# img_result = []
# img_for_class = img.copy()
# margin_pixel = 60
#
# for rect in rects:
#     img_result.append(
#         img_for_class[rect[1] - margin_pixel: rect[1] + rect[3] + margin_pixel,
#         rect[0] - margin_pixel: rect[0] + rect[2] + margin_pixel])
#     # Draw the rectangles
#     cv2.rectangle(img, (rect[0], rect[1]),
#                   (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5)
# plt.figure(figsize=(15,12))
# plt.imshow(cv2.resize(img_result[0], (28,28)))
# plt.show()
