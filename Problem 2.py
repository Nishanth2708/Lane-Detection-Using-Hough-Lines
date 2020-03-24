import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import math


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
        Note: if threshold isn't done properly the image probably results to noise'''

    # bitwise=Canny(bitwise)
    # lower_white=[0,0,255]
    # upper_white=[255,255,255]

    x1=200
    y1=800
    x2=50
    y2=500

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

    # cv2.imshow('jhsdgfdgsh',img_bwise)

    rho=2
    theta=np.pi/60

    line = cv2.HoughLinesP(img_bwise, rho, theta, 110, np.array([]), minLineLength=40, maxLineGap=25)

    return line


def draw_lines(img1, lines, color=[0, 0, 255], thickness=10):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.

    img = np.copy(img1)
    # Create a blank image that matches the original in size.
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    # Loop over all lines and draw them on the blank image.
    # print(line_img)
    # print(img.shape[0],img.shape[1])
    # print(lines)

    # x1=200
    # y1=800
    # x2=50
    # y2=500

    for line in lines:

        # print(line)

        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]

        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # if (x1, y1, x2, y2) is not None:
    #
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img

# def each_line(frame,lines):
#
#     left_lane_x=[]
#     right_lane_x=[]
#     left_lane_y=[]
#     right_lane_y=[]
#     for line in lines:
#         for a,b,c,d in line:
#
#             slope=(d-b)/(c-a)
#             if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
#                 continue
#             if slope < 0:
#                 left_lane_x.extend([a,c])
#                 left_lane_y.extend([b,d])
#             else:
#
#                 right_lane_x.extend([a,c])
#                 right_lane_y.extend([b,d])
#
#     min_y=frame.shape[0]*0.6
#     max_y=frame.shape[0]
#
#     poly_left = np.polyfit(left_lane_y,left_lane_x,deg=1)
#     poly_right = np.polyfit(right_lane_y,right_lane_x,deg=1)
#
#     right_x_start = int(poly_right(max_y))
#     right_x_end = int(poly_right(min_y))
#
#     left_x_start = int(poly_left(max_y))
#     left_x_end = int(poly_left(min_y))
#
#     line_img= draw_lines(frame,[[[left_x_start,max_y,left_x_end,min_y],[right_x_start,max_y,right_x_end,min_y]]],thickness=7)
#
#     return line_img






#Camera Matrix
k=np.array([[9.037596e+02,0.000000e+00, 6.957519e+02],[0.000000e+00, 9.019653e+02, 2.242509e+02],
           [0.000000e+00, 0.000000e+00, 1.000000e+00]])

# Distortion Coeefficients
D=np.array([ -3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

# img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

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

# Uncomment only if you want to select different points for estimating homoography
while True:
    print('Double Click to store new points; Otherwise PRESS ESC')
    print('Select point in clockwise manner starting from top-left')
    # cv2.imshow('Finding the Four Points', img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    break



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

cap=cv2.VideoCapture('project.avi')

count = 0

while (cap.isOpened()):
    ret,frame=cap.read()

    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # k=frame.shape[:2]
    # print(k)

    if frame is None:
        continue
    else:
        w,h=frame.shape[:2]

    frame=cv2.resize(frame,(640,480))
    count+=1
    print(count)

    # slope=(y2-y1)/(x2-x1)
    #
    #             if slope < 0:
    #                 Left_Lane.append(x1,y1,x2,y2,slope)
    #
    #             else:
    #                 Right_Lane.append(x1,y1,x2,y2,slope)



    # a = Hough_Transform(frame)
    # # if a is None:
    # #     continue
    # # else:
    # #     # print(a[:,0])
    # for line in range(0,len(a)):
    #     # print(line)
    #     for b,c,d,e in a[line]:
    #         print(b,c,d,e)
    test=undist(frame,k,D)
    # test2=crop(test)
    # test1=Canny(frame)
    a=Hough_Transform(frame)

    if a is not None:

        test3=draw_lines(frame,a)

    else:
        test3 = frame



    cv2.imshow('homo',test3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # else:
    #     break

# When everything done, release the capture



# print('finished')
cap.release()
cv2.destroyAllWindows()
















