# openCV contour rotation and scale about either center of mass of arbitrary point
# https://math.stackexchange.com/questions/3245481/rotate-and-scale-a-point-around-different-origins
# i wanted to scale up a contour w/o having to use images and morphological dilate. 

import numpy as np
import cv2

def scaleRotateContourAboutPoint(contour, scale_xy, angle_deg, pivot = None):
    """
    In order to rotate and set of points around arbitrary point you can create a transformation matrix M.
    Rotation/scaling around non-origin point requires that rotation pivot is translated to the origin.
    This is done by applying matrix MTcenter. Points can be scaled and rotated using matrices MRotate & MScale.
    After, rotated points are translated to their original pivot point
    NOTE: in image space (y faces down), rotation is applied clockwise.
    NOTE: if pivot = None, rotation will be applied w.r.t centre of mass
    """
    OG_shape, OG_type = contour.shape, contour.dtype    # return same type of object

    ones        = np.ones((OG_shape[0], 1))              # need to pad 1 for translation
    points      = contour.reshape(-1,2)                 # reshape from N,1,2 to N,2
    points      = np.hstack((points,ones))              # pad

    a = angle_deg*np.pi/180

    if pivot is None: 
        (cx, cy) = (lambda m: (m['m10']/m['m00'], m['m01']/m['m00'])) (cv2.moments(contour))
    else:
        (cx, cy) = pivot

    (sx, sy)  = scale_xy

    MTcenter = np.array([
                            [1, 0, -cx],
                            [0, 1, -cy],
                            [0, 0,   1]     ])
    
    MRotate = np.array([
                            [np.cos(a), -np.sin(a), 0],
                            [np.sin(a),  np.cos(a), 0],
                            [0        ,  0        , 1]    ])
    MScale = np.array([
                            [sx, 0 , 0],
                            [0 , sy, 0],
                            [0 , 0 , 1] ])
    
    MTback = np.array([ 
                            [1, 0, cx],
                            [0, 1, cy],
                            [0, 0, 1 ]  ])

    M  = np.linalg.multi_dot([MTback,MScale,MRotate,MTcenter])

    points_remap = M @  points.T

    points_remap = points_remap[:2].T.reshape(OG_shape).astype(OG_type)

    return points_remap, np.uint32((cx,cy))



image = np.zeros((150,150), np.uint8)  

rect_xy = (50, 30)
rect_wh = (100, 120)

cv2.rectangle(image, rect_xy, rect_wh, 255, -1)

contour = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]

(sx,sy) = (1,1) 
angle = -15 

customPivot = rect_xy

cntr, (cx,cy) = scaleRotateContourAboutPoint(contour = contour, scale_xy = (sx,sy), angle_deg = angle, pivot = customPivot)

cv2.drawContours( image, [cntr], -1, 175, 2) 
cv2.circle(image,(cx,cy), 5, 120, -1)
cv2.imshow('anis-scale + rot about pivot', image) 

k = cv2.waitKey(0)
if k == 27:  # close on ESC key
    cv2.destroyAllWindows()

