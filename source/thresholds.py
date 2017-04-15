import cv2
import numpy


def threshold_pipeline(image):
    yellow_binary = color_threshold(image, h_thresh=(0, 50), s_thresh=(90, 255))
    white_binary = color_threshold(image, s_thresh=(0, 30), v_thresh=(200, 255))
    hsv_result = cv2.bitwise_or(yellow_binary, white_binary)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    gradient_combined = gradient_pipeline(l) + gradient_pipeline(s)
    gradient_combined_blur = cv2.GaussianBlur(gradient_combined, (5, 5), 0)
    hls_result = numpy.zeros_like(gradient_combined_blur)
    hls_result[gradient_combined_blur > 0.5] = 1

    combined_thresholds = cv2.bitwise_or(hsv_result, hls_result)
    return combined_thresholds


def gradient_pipeline(image):
    magnitude = magnitude_of_gradient(image, (20, 150))
    direction = direction_of_gradient(image, (.6, 1.1))
    gradient_x = gradient(image, 'x', (50, 200))
    gradient_y = gradient(image, 'y', (50, 200))
    result = numpy.zeros_like(direction)
    result[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
    return result


def color_threshold(image, h_thresh=(0, 255), s_thresh=(0, 255), v_thresh=(0, 255)):
    image = numpy.copy(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(numpy.float)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    result = numpy.zeros_like(s_channel)
    result[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]))
           & ((h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]))
           & ((v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]))] = 1
    return result


def magnitude_of_gradient(image, thresholds=(0, 255)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    gradient_magnitude = numpy.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scale_factor = numpy.max(gradient_magnitude) / 255
    gradient_magnitude = (gradient_magnitude / scale_factor).astype(numpy.uint8)
    result = numpy.zeros_like(gradient_magnitude)
    result[(gradient_magnitude >= thresholds[0]) & (gradient_magnitude <= thresholds[1])] = 1
    return result


def direction_of_gradient(image, thresholds=(0, numpy.pi / 2)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    absolute_gradient_direction = numpy.arctan2(numpy.absolute(sobel_y), numpy.absolute(sobel_x))
    result = numpy.zeros_like(absolute_gradient_direction)
    result[(absolute_gradient_direction >= thresholds[0]) & (absolute_gradient_direction <= thresholds[1])] = 1
    return result


def gradient(image, orientation='x', thresholds=(0, 255)):
    image = numpy.copy(image)
    sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    if orientation == 'y':
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    absolute_sobel = numpy.absolute(sobel)
    scaled_sobel = numpy.uint8(255 * absolute_sobel / numpy.max(absolute_sobel))
    result = numpy.zeros_like(scaled_sobel)
    result[(scaled_sobel >= thresholds[0]) & (scaled_sobel <= thresholds[1])] = 1
    return result
