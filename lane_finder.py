import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from PIL import Image

#define vars for indices
SLOPE=0
X1=4
Y1=5
X2=6
Y2=7
INTERCEPT1=1
INTERCEPT2=2
LENGTH=3

#define poly coords
POLY_LEFT_BOT=(160, 540)
POLY_LEFT_TOP=(420,330)
POLY_RIGHT_TOP=(520,330)
POLY_RIGHT_BOT=(880,540)

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


def draw_side(img, lines, color=[255,0,0], thickness=2):
    #figure out which which top corner to use for distance metrics and 
    #which point references the top of the line segment
    xtop=X1;ytop=Y1;xbot=X2;ybot=Y2
    top = POLY_RIGHT_TOP
    if lines[0][SLOPE] <= 0.0:
        top=POLY_LEFT_TOP
        xtop=X2;ytop=Y2;xbot=X1;ybot=Y1

    #find topmost point
    start_points = (int(lines[0][xtop]), int(lines[0][ytop]))
    top_length = np.linalg.norm(np.array(start_points) - np.array(top))
    i=1
    while i < len(lines):
        lines[i][LENGTH] = np.linalg.norm(np.array((int(lines[i][xtop]), int(lines[i][ytop]))) - np.array(top))
        if lines[i][LENGTH] < top_length:
            top_length = lines[i][LENGTH]
            start_points = (int(lines[i][xtop]), int(lines[i][ytop]))
        i+=1
    #draw the line using the mean avg slope/intercept along with known y-coord of bottom frame (to find x)
    # start from top most line segment
    mean_slope = np.sum(lines[:,SLOPE])/len(lines)
    mean_intercept = np.sum(lines[:,INTERCEPT1])/len(lines)
    x = int((-1 * (mean_intercept/mean_slope)) + (POLY_RIGHT_BOT[1]/mean_slope))
#    cv2.line(img, start_points, (x, POLY_RIGHT_BOT[1]), color, thickness)

    #also draw using median (for comparison)
    median_slope = lines[np.argsort(lines[:,SLOPE])][int(len(lines)/2),SLOPE]
    median_intercept = lines[np.argsort(lines[:,INTERCEPT1])][int(len(lines)/2),INTERCEPT1]
    x = int((-1 * (median_intercept/median_slope)) + (POLY_RIGHT_BOT[1]/median_slope))
    cv2.line(img, start_points, (x, POLY_RIGHT_BOT[1]), color, thickness)

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
    lines_eq=np.zeros((len(lines), 8))
    for i, line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            lines_eq[i][SLOPE] = slope
            lines_eq[i][X1:] = np.copy(line)
            lines_eq[i][INTERCEPT1] = y1 - slope * x1
            lines_eq[i][INTERCEPT2] = y2 - slope * x2

    #sort the array by slope
    sides = lines_eq[np.argsort(lines_eq[:,SLOPE])]

    #now, advance in sorted sides until you reach a slope that belongs to left
    #side of the lane (right side will be negative...thus at beginning)
    right_idx=1     
    while right_idx < len(sides) and sides[right_idx][SLOPE] <= 0.0:
        right_idx += 1

    #draw sides
    left_line = np.copy(sides[:right_idx])
    draw_side(img, left_line, color, 10)

    right_line = np.copy(sides[right_idx:])
    draw_side(img,right_line, color, 10)

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
#        file = 'whiteCarLaneSwitch.jpg'
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
        vertices = np.array([[POLY_LEFT_BOT, POLY_LEFT_TOP, POLY_RIGHT_TOP, POLY_RIGHT_BOT]], dtype=np.int32)

        # apply vertices as an image mask to the image
        masked_edges = region_of_interest(edges, vertices)

        #convert masked image to an image with hough lines drawn
        image_lines = hough_lines(masked_edges, 2, np.pi/180, 20, 25, 15)

        #draw the lines on the original image
        image = weighted_img( image_lines, image )

        image_to_save = Image.fromarray(image)
        image_to_save.save('test_images_output/{}'.format(file))

        image_to_show = image
        plt.imshow(image_to_show, cmap='gray')
        plt.show()
