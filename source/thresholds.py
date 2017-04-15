import cv2
import numpy


def gradient(image, orient='x', sobel_kernel=3, thresholds=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    absolute_sobel = numpy.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        absolute_sobel = numpy.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = numpy.uint8(255 * absolute_sobel / numpy.max(absolute_sobel))
    result = numpy.zeros_like(scaled_sobel)
    result[(scaled_sobel >= thresholds[0]) & (scaled_sobel <= thresholds[1])] = 1
    return result


def magnitude_of_gradient(image, sobel_kernel=3, thresholds=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradient_magnitude = numpy.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scale_factor = numpy.max(gradient_magnitude) / 255
    gradient_magnitude = (gradient_magnitude / scale_factor).astype(numpy.uint8)
    result = numpy.zeros_like(gradient_magnitude)
    result[(gradient_magnitude >= thresholds[0]) & (gradient_magnitude <= thresholds[1])] = 1
    return result


def direction_of_gradient(image, sobel_kernel=3, thresholds=(0, numpy.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absolute_gradient_direction = numpy.arctan2(numpy.absolute(sobel_y), numpy.absolute(sobel_x))
    result = numpy.zeros_like(absolute_gradient_direction)
    result[(absolute_gradient_direction >= thresholds[0]) & (absolute_gradient_direction <= thresholds[1])] = 1
    return result


def color_threshold(image, thresholds_s=(0, 255), thresholds_v=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    hls_result = numpy.zeros_like(s_channel)
    hls_result[(s_channel > thresholds_s[0]) & (s_channel <= thresholds_s[1])] = 1
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    hsv_result = numpy.zeros_like(v_channel)
    hsv_result[(v_channel > thresholds_v[0]) & (v_channel <= thresholds_v[1])] = 1
    result = numpy.zeros_like(s_channel)
    result[(hls_result == 1) & (hsv_result == 1)] = 1
    return hls_result


def threshold_pipeline(image):
    gradient_x = gradient(image, thresholds=(12, 255))
    gradient_y = gradient(image, orient='y', thresholds=(25, 255))
    # magnitude = magnitude_of_gradient(image, thresholds=(30, 100))
    # direction = direction_of_gradient(image, 15, (0.7, 1.3))
    color = color_threshold(image, (140, 255), (50, 255))
    combined_thresholds = numpy.zeros_like(image[:, :, 0])
    gradient_hits = (gradient_x == 1) & (gradient_y == 1)
    # magnitude_direction = (magnitude == 1) & (direction == 1)
    # combined_thresholds[gradient_hits | magnitude_direction | color == 1] = 255
    combined_thresholds[gradient_hits | color == 1] = 255
    return combined_thresholds
