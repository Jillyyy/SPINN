import cv2

img = cv2.imread('/project/SPINN/test_kp.jpg')

print(img.shape)
# img_c = cv2.circle(img, (100, 60), 3, (0, 0, 213), -1)

# cv2.imwrite('test.jpg', img_c)