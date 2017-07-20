import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from PIL import Image

#define vars for indices
SLOPE=0
X1=1
Y1=2
X2=3
Y2=4

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_down(img, lines, bottom, dist=6, color=[255,0,0], thickness=2):
    points = np.zeros((len(lines), 5), dtype=np.int32)
    xtop=1;ytop=2;xbot=3;ybot=4
    is_left = False
    if bottom[0] == 0:
        is_left = True
        print('\n========\nIn LEFT Line')
        xtop=3;ytop=4;xbot=1;ybot=2
    else:
        print('\n========\nIn RIGHT Line')
    #get distances in order to sort points properly
    for i, line in enumerate(lines):
        length = np.linalg.norm(np.array([line[xtop],line[ytop]]) - np.array(bottom[0], bottom[1]))
        points[i][0] = length
        points[i][1:] = np.copy(line[1:])
    dist_sort = points[np.argsort(points[:,0])]
    slope = lines[int(len(lines)/2)][SLOPE]

    #draw down the array. if not near x==0, finish by calculating the line segments with slope
    print('SLOPE SORT:\t',lines)
    print('DIST_SORT:\t',dist_sort)
    i = len(dist_sort)-1
    while i > 0:
        x1 = dist_sort[i][xtop]
        y1 = dist_sort[i][ytop]
        x2 = dist_sort[i-1][xtop]
        y2 = dist_sort[i-1][ytop]
        print('Drawing: ({}, {}), ({}, {})'.format(x1, y1, x2, y2))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        i-=1

    #draw last line
    x1 = dist_sort[0][xtop]
    y1 = dist_sort[0][ytop]
    x2 = dist_sort[0][xbot]
    y2 = dist_sort[0][ybot]
    print('Drawing Last Line: ({}, {}), ({}, {})'.format(x1, y1, x2, y2))
    cv2.line(img, (x1,y1), (x2,y2), [0,0,255], thickness)

    #is there a gap at the bottom? fill it in
    x1 = x2
    y1 = y2
    while True:  
        x2 = int(x1+dist*np.sin(slope))
        y2 = int(y1+dist*np.cos(slope))
        if y2 >= bottom[1]:
            break
        if (is_left and x2 <= bottom[0]) or (not is_left and x2 >= bottom[0]):
            break
        print('Drawing Bottom Gap: ({}, {}), ({}, {})'.format(x1, y1, x2, y2))
        cv2.line(img, (x1,y1), (x2,y2), [0,255,0], thickness)
        x1 = x2
        y1 = y2        


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    #calculate the slope of each line segment in order to figure out which side
    #the segment belongs
    THRESHOLD=0.2
    lines_with_slope=np.zeros((len(lines), 5))
    for i, line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            lines_with_slope[i][0] = slope
            lines_with_slope[i][1:] = np.copy(line)
            cv2.line(img, (x1, y1), (x2,y2), color, thickness)
#            length = np.linalg.norm(np.array([x2,y2]) - np.array([x1,y1]))

    #sort the array by slope
    sides = lines_with_slope[np.argsort(lines_with_slope[:,0])]

    #now, advance in sorted sides until you reach a slope that belongs to left
    #side of the lane (right side will be negative...thus at beginning)
    prev = sides[0]
    right_idx=1     
    while right_idx < len(sides) and (sides[right_idx][SLOPE]-prev[SLOPE]) <= THRESHOLD:
        prev = sides[right_idx]
        right_idx += 1


    #draw sides
    LEFT_BOTTOM = np.array((0,img.shape[0]))
    left_line = np.copy(sides[:right_idx])
    draw_down(img, left_line, LEFT_BOTTOM, 10, color, 10)

    RIGHT_BOTTOM = np.array((img.shape[1],img.shape[0]))
    right_line = np.copy(sides[right_idx:])
    draw_down(img,right_line, RIGHT_BOTTOM, 10, color, 10)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    #getting line segments using the probabilistic Hough transform
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

if __name__ == '__main__':
    if not os.path.exists('test_images_output'):
        os.makedirs('test_images_output')
    for file in os.listdir('test_images'):
        #read in image
        image = mpimg.imread('test_images/{}'.format(file))

        #convert image to grayscale
        gray_image = grayscale(image)

        #smooth image
        blur_gray_image = gaussian_blur(gray_image, 3)

        #apply the canny transform for edge detection
        edges = canny(blur_gray_image, 50, 150)

        #define a polygon to use as the image mask
        imshape = blur_gray_image.shape
        vertices = np.array([[(80, imshape[0]), (421,330), (521, 330), (900, imshape[0])]], dtype=np.int32)

        # apply vertices as an image mask to the image
        masked_edges = region_of_interest(edges, vertices)

        #convert masked image to an image with hough lines drawn
        image_lines = hough_lines(masked_edges, 2, np.pi/180, 20, 25, 15)

        #draw the lines on the original image
        image = weighted_img( image_lines, image )

        image_to_save = Image.fromarray(image)
        image_to_save.save('test_images_output/{}'.format(file))
        #cv2.imwrite('test_images_output/{}'.format(file), image)
        plt.imshow(image)
        plt.show()
