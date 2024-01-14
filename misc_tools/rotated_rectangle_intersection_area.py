import cv2, numpy as np
from matplotlib import pyplot as plt
"""
detect if rectangles are intsecting and calculate intersection area
"""
# two rectangles with area of 100
xywh1 = [10,  5, 10, 10]
xywh2 = [10, 15, 10, 10]

blank = np.zeros((30,30), np.uint8)

to_rotated = lambda x,y,w,h,a: ((x + w//2, y + h//2), (w, h), a)
rct_pts = lambda rect: np.intp(cv2.boxPoints(rect))

r1 = to_rotated(*xywh1, 45)
r2 = to_rotated(*xywh2, 45)

r1_pts = rct_pts(r1)
r2_pts = rct_pts(r2)

cv2.drawContours(blank,[r1_pts], 0, 200, 1)
cv2.drawContours(blank,[r2_pts], 0, 128, 1)

plt.imshow(blank)

interType, points  = cv2.rotatedRectangleIntersection(r1, r2)

if interType > 0:
    intersection_area = cv2.contourArea( np.array(points, dtype=np.int32)) 
    print(intersection_area)

plt.show()
