import cv2

origin_img = cv2.imread('./images/auto_zoom/swan-3531856_1280.jpg')
blurred_img_1 = cv2.blur(origin_img, ksize=(22, 22))
blurred_img_2 = cv2.blur(origin_img, ksize=(11, 11))
cv2.imwrite("./images/auto_zoom/swan_blurred_1.jpg", blurred_img_1)
cv2.imwrite("./images/auto_zoom/swan_blurred_2.jpg", blurred_img_2)
