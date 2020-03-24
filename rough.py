import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import math

source_points = []
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        print(mouseX, mouseY)
        source_points.append([mouseX,mouseY])

img=cv2.imread(r'C:\Users\12027\PyCharm_Projects\Lane_Detection\data\0000000000.png')
# img=cv2.resize(img,[640,480])
cv2.namedWindow('Finding the Four Points')
cv2.setMouseCallback('Finding the Four Points', draw_circle)

'''Uncomment to generate Video from the images'''
# img_array = []
# for filename in glob.glob(r'C:\Users\12027\PyCharm_Projects\Lane_Detection\data\*.png'):
#     img = cv2.imread(filename)
#
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
#     print(img)
# out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

# Uncomment only if you want to select different points for estimating homoography
while True:
    print('Double Click to store new points; Otherwise PRESS ESC')
    print('Select point in clockwise manner starting from top-left')
    # cv2.imshow('Finding the Four Points', img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    break

def Homography(img):
    # img=cv2.resize(img,[640,480])
    src = np.array([[518, 280], [649, 300], [680, 456], [163, 418]])
    dst = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype='f')

    H, status = cv2.findHomography(src, dst)
    warp = cv2.warpPerspective(img, H, (640, 480))
    # img_rotate = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_rotat2 = cv2.rotate(img_rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return warp

def undist(img,CameraMat,K):
    CameraMat=np.copy(CameraMat)
    K=np.copy(K)
    undistorted=cv2.undistort(img,CameraMat,K)
    return undistorted

def denoise(img,Kernel=(7,7)):
    # This shoudld be changed based on the intensity of noise
    # img=cv2.resize(img,(640,480))

    GaussBlur=cv2.GaussianBlur(img,Kernel,0)

    return GaussBlur

def crop(img):

    '''Tweak the Values of the
    X1,X2,X3 AND X4 to extract the ROI'''
    # img=cv2.resize(img,(640,480))
    x1=200
    y1=800
    x2=50
    y2=500

    roi= img[x1:y1, x2:y2]

    return roi

def Canny (img_C):
    # img=cv2.resize(img,(640,480))
    # img=crop(img)
    min_val=400
    max_val=600
    edge_=cv2.Canny(img_C,min_val,max_val)

    return edge_

############################ Step 2 ############################


def mask(img_):
    '''To define the mask first consider a blank image
        Define the Threshold for the white pixels by converting to HSV OR HLS
        Bitwise and Detects the white pixels on the image
        Note: if thresholding isn't done properly the image probably results to noise'''

    # bitwise=Canny(bitwise)
    # lower_white=[0,0,255]
    # upper_white=[255,255,255]

    # roi= img_[x1:y1, x2:y2]

    # cv2.imshow('ssd',roi)


    ### Applying segementation onto the image

    # img=cv2.resize(img,(640,480))

    img_bgr=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

    mask=cv2.inRange(img_bgr,220,255)
    mask=denoise(mask)

    mask_canny =Canny(mask)


    bitwise=cv2.bitwise_and(img_bgr,img_bgr,mask=mask_canny)
    # print(bitwise)

    return bitwise

def Hough_Transform(img_bwise):
    img_bwise=mask(img_bwise)


    rho=3
    theta=np.pi/180

    line = cv2.HoughLinesP(img_bwise, rho, theta, 120, np.array([]), minLineLength=50, maxLineGap=25)

    return line


def draw_lines(img1, lines, color=[0, 255, 255], thickness=8):


    img = np.copy(img1)

    imshape = img1.shape
    ymin_global = img.shape[0]
    ymax_global = img1.shape[0]

    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.

    # Create a blank image that matches the original in size.
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

    slope_threshold=0.5
    left_lines_x=[]
    right_lines_x=[]
    left_lines_y=[]

    right_lines_y=[]
    left_slope   = []
    right_slope  =[]

    for line in lines:

        for x1,y1,x2,y2 in line:


            slopes,intercept=np.polyfit((x1,x2),(y1,y2),1) # 1 Degree polynomial y=m*x+b
            ymin_global = min(min(y1, y2), ymin_global)

            if slopes==math.inf:
                slopes=999

            elif slopes==0:
                continue


            if slopes > slope_threshold:

                left_slope += [slopes]
                left_lines_x += [x1,x2]
                left_lines_y += [y1,y2]

            else:

                right_slope += [slopes]
                right_lines_x += [x1,x2]
                right_lines_y += [y1,y2]

            left_slope_mean=np.mean(left_slope)
            left_lines_x_mean=np.mean(left_lines_x)
            left_lines_y_mean=np.mean(left_lines_y)
            left_intercept=(left_lines_y_mean) - (left_slope_mean*left_lines_x_mean)

            right_slope_mean=  np.mean(right_slope)
            right_lines_x_mean= np.mean(right_lines_x)
            right_lines_y_mean=np.mean(right_lines_y)
            right_intercept=(right_lines_y_mean) - (right_slope_mean*right_lines_x_mean)

            if ((len(left_slope) > 0) and (len(right_slope) > 0)):
                upper_left_x = int((ymin_global - left_intercept) / left_slope_mean)
                lower_left_x = int((ymax_global - left_intercept) / left_slope_mean)
                upper_right_x = int((ymin_global - right_intercept) / right_slope_mean)
                lower_right_x = int((ymax_global - right_intercept) / right_slope_mean)

                cv2.line(img, (upper_left_x, ymin_global),
                         (lower_left_x, ymax_global), color, thickness)
                cv2.line(img, (upper_right_x, ymin_global),
                         (lower_right_x, ymax_global), color, thickness)




    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(line_img, 0.8, img, 1.0, 0.0)


    return img





#Camera Matrix
k=np.array([[9.037596e+02,0.000000e+00, 6.957519e+02],[0.000000e+00, 9.019653e+02, 2.242509e+02],
           [0.000000e+00, 0.000000e+00, 1.000000e+00]])

# Distortion Coeefficients
D=np.array([ -3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])



cap=cv2.VideoCapture('project.avi')

count = 0

while (cap.isOpened()):
    ret,frame=cap.read()



    if frame is None:
        continue
    else:
        w,h=frame.shape[:2]

    frame=cv2.resize(frame,(640,480))
    count+=1
    print(count)
    test=undist(frame,k,D)

    a=Hough_Transform(frame)

    if a is not None:

        test3=draw_lines(frame,a)

    else:
        test3 = frame



    cv2.imshow('homo',test3)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
















