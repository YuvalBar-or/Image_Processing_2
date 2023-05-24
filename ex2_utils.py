import math
import numpy as np
import cv2
from ex2_utils import *



def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 214329633


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    kernel_length = len(k_size)
    signal = np.pad (in_signal , (kernel_length-1 , kernel_length -1) , 'constant')
    signal_len = len(signal)
    con_signal = np.zeros(signal_len-kernel_length+1)
    for i in range (len(con_signal)):
        con_signal[i] = (signal[ i:i + kernel_length]* k_size).sum()
    return con_signal


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    kernel = np.flip(kernel)
    img_height, img_w = in_image.shape
    ker_h, ker_w = kernel.shape
    padded_img = np.pad(in_image, (ker_h // 2, ker_w // 2), 'edge')
    returned_pic = np.zeros((img_height, img_w))
    for y in range(img_height):
        for x in range(img_w):
            returned_pic[y, x] = (padded_img[y:y + ker_h, x:x + ker_w] * kernel).sum()
    return returned_pic


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    X_axis = np.array([[0, 0, 0],
                       [-1, 0, 1],
                       [0, 0, 0]]
                      )
    X_output = conv2D(in_image, X_axis)
    Y_output = conv2D(in_image, X_axis.transpose())
    angle_direction = np.arctan2(X_output, Y_output)
    magnitude = np.sqrt(np.square(X_output) + np.square(Y_output))
    return angle_direction, magnitude


def create_gaussian_kernel(kernel, sigma=1) -> np.ndarray:
  kernel = int(kernel) // 2
  x, y = np.mgrid[-kernel:kernel + 1, -kernel:kernel + 1]
  normal = 1 / (2.0 * np.pi * sigma**2)
  gaussian = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
  return gaussian


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = create_gaussian_kernel(k_size, sigma)
    blurred = conv2D(in_image, kernel)
    return blurred


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, 0)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    laplacian_ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img = conv2D(img, laplacian_ker)
    zero_crossing = np.zeros(img.shape)
    for i in range(img.shape[0] - (laplacian_ker.shape[0] - 1)):
        for j in range(img.shape[1] - (laplacian_ker.shape[1] - 1)):
            if img[i][j] == 0:
                if (img[i][j - 1] < 0 and img[i][j + 1] > 0) or \
                        (img[i][j - 1] < 0 and img[i][j + 1] < 0) or \
                        (img[i - 1][j] < 0 and img[i + 1][j] > 0) or \
                        (img[i - 1][j] > 0 and img[i + 1][j] < 0):  # All his neighbors
                    zero_crossing[i][j] = 255
            if img[i][j] < 0:
                if (img[i][j - 1] > 0) or (img[i][j + 1] > 0) or (img[i - 1][j] > 0) or (img[i + 1][j] > 0):
                    zero_crossing[i][j] = 255
    return zero_crossing


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(img_gaussian)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    def preprocess_image(img):
        processed_image = (img * 255).astype(np.uint8)
        return processed_image

    def detect_edges(img):
        edge_detected_image = cv2.Canny(img, 100, 200)
        return edge_detected_image

    def generate_circle_points(lower_radius, upper_radius, number_of_steps):
        circle_points = []
        for rad in range(lower_radius, upper_radius + 1):
            for step in range(number_of_steps):
                angle = (step / number_of_steps) * 2 * np.pi
                x_cord = int(np.cos(angle) * rad)
                y_cord = int(np.sin(angle) * rad)
                circle_points.append((x_cord, y_cord, rad))
        return circle_points

    def extract_edge_pixels(edge_detected_image):
        edge_locations = []
        num_rows, num_cols = edge_detected_image.shape
        for x in range(num_rows):
            for y in range(num_cols):
                if edge_detected_image[x, y] == 255:
                    edge_locations.append((y, x))
        return edge_locations

    def tally_circles(edge_locations, circle_points):
        circle_dictionary = {}
        for x1, y1 in edge_locations:
            for x2, y2, rad in circle_points:
                dx, dy = x1 - x2, y1 - y2
                circle_key = circle_dictionary.get((dx, dy, rad))
                if circle_key is None:
                    circle_dictionary[(dx, dy, rad)] = 1
                else:
                    circle_dictionary[(dx, dy, rad)] += 1
        return circle_dictionary

    def filter_results(circle_dictionary, number_of_steps, threshold_ratio):
        final_result = []
        for circle, count in sorted(circle_dictionary.items(), key=lambda v: -v[1]):
            nx, ny, rad = circle

            if count / number_of_steps >= threshold_ratio and all(
                    (nx - x) ** 2 + (ny - y) ** 2 > r ** 2 for x, y, r in final_result):
                final_result.append((nx, ny, rad))
        return final_result

    processed_image = preprocess_image(img)
    edge_detected_image = detect_edges(processed_image)
    circle_points = generate_circle_points(min_radius, max_radius, 100)
    edge_pixels = extract_edge_pixels(edge_detected_image)
    circle_counts = tally_circles(edge_pixels, circle_points)
    final_result = filter_results(circle_counts, 100, 0.5)
    return final_result





def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    final_image = np.zeros_like(in_image)
    filter = cv2.bilateralFilter(in_image, k_size,sigma_space, sigma_color )
    k_size = int(k_size / 2)
    in_image = cv2.copyMakeBorder(in_image, k_size, k_size, k_size, k_size,
                                  cv2.BORDER_REPLICATE, None, value=0)
    for y in range(k_size, in_image.shape[0] - k_size):
        for x in range(k_size, in_image.shape[1] - k_size):
            # next few lines is the official formula given in class
            pivot = in_image[y, x]
            neighborhood = in_image[
                            y - k_size:y + k_size + 1,
                            x - k_size:x + k_size + 1
                            ]
            delta = neighborhood.astype(int) - pivot
            gaussian = np.exp(-np.power(delta, 2) / (2 * sigma_space))
            get_gaussian = cv2.getGaussianKernel(2 * k_size + 1, sigma=sigma_color)
            get_gaussian = get_gaussian.dot(get_gaussian.T)
            final_gaussian = get_gaussian * gaussian

            final_sum = ((final_gaussian * neighborhood / final_gaussian.sum()).sum())

            final_image[y - k_size, x - k_size] = round(final_sum)
    return filter, final_image
