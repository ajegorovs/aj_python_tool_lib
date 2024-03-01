import cv2, numpy as np
#import matplotlib.pyplot as plt

blue = (0,0,255)
red  = (255,0,0)
green = (0,0,255)
white = (255,255,255)
black = (0,0,0)

def hconcat_del(arr, del_width = 5, del_clr = None, bw = 5):
    """
    modify cv2.hconcat to include vertical delimeter between images.
    """
   
    num_ch = len(arr[0].shape)
    if del_clr == None:
        if num_ch == 3:
            del_clr = [0,0,0]
        else:
            del_clr = 0

    shape = arr[0][:,:del_width].shape
    delimeter = np.full(shape = shape, fill_value=del_clr, dtype = np.uint8)

    #insert delimeter between images
    out_arr = []
    for img in arr:
        out_arr.append(img)
        out_arr.append(delimeter)

    out1 = cv2.hconcat(out_arr[:-1])
    # add border by creating a padded image and inserting inside our concat
    border_off = np.zeros(num_ch, int)
    border_off[:2] = 2*bw
    shape2 = np.array(out1.shape) + border_off
    blank = np.full(shape2, del_clr)
    blank[bw:-bw,bw:-bw] = out1
    return blank