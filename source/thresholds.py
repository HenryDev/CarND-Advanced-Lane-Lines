import cv2
import numpy


# def gradient(image, orient='x', sobel_kernel=3, thresholds=(0, 255)):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     absolute_sobel = numpy.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
#     if orient == 'y':
#         absolute_sobel = numpy.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
#     scaled_sobel = numpy.uint8(255 * absolute_sobel / numpy.max(absolute_sobel))
#     result = numpy.zeros_like(scaled_sobel)
#     result[(scaled_sobel >= thresholds[0]) & (scaled_sobel <= thresholds[1])] = 1
#     return result
#
#
# def magnitude_of_gradient(image, sobel_kernel=3, thresholds=(0, 255)):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     gradient_magnitude = numpy.sqrt(sobel_x ** 2 + sobel_y ** 2)
#     scale_factor = numpy.max(gradient_magnitude) / 255
#     gradient_magnitude = (gradient_magnitude / scale_factor).astype(numpy.uint8)
#     result = numpy.zeros_like(gradient_magnitude)
#     result[(gradient_magnitude >= thresholds[0]) & (gradient_magnitude <= thresholds[1])] = 1
#     return result
#
#
# def direction_of_gradient(image, sobel_kernel=3, thresholds=(0, numpy.pi / 2)):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
#     absolute_gradient_direction = numpy.arctan2(numpy.absolute(sobel_y), numpy.absolute(sobel_x))
#     result = numpy.zeros_like(absolute_gradient_direction)
#     result[(absolute_gradient_direction >= thresholds[0]) & (absolute_gradient_direction <= thresholds[1])] = 1
#     return result
#
#
# def color_threshold(image, thresholds_s=(0, 255), thresholds_v=(0, 255)):
#     hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#     s_channel = hls[:, :, 2]
#     hls_result = numpy.zeros_like(s_channel)
#     hls_result[(s_channel > thresholds_s[0]) & (s_channel <= thresholds_s[1])] = 1
#     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     v_channel = hsv[:, :, 2]
#     hsv_result = numpy.zeros_like(v_channel)
#     hsv_result[(v_channel > thresholds_v[0]) & (v_channel <= thresholds_v[1])] = 1
#     result = numpy.zeros_like(s_channel)
#     result[(hls_result == 1) & (hsv_result == 1)] = 1
#     return hls_result


# def threshold_pipeline(image):
#     gradient_x = gradient(image, thresholds=(12, 255))
#     gradient_y = gradient(image, orient='y', thresholds=(25, 255))
#     magnitude = magnitude_of_gradient(image, thresholds=(30, 100))
#     direction = direction_of_gradient(image, 15, (0.7, 1.3))
#     color = color_threshold(image, (140, 255), (50, 255))
#     combined_thresholds = numpy.zeros_like(image[:, :, 0])
#     gradient_hits = (gradient_x == 1) & (gradient_y == 1)
#     # magnitude_direction = (magnitude == 1) & (direction == 1)
#     # combined_thresholds[gradient_hits | magnitude_direction | color == 1] = 255
#     combined_thresholds[gradient_hits | color == 1] = 255
#     return combined_thresholds

def threshold_pipeline(image):
    # Extract yellow binary
    yellow_binary = color_threshold(image,
                                    h_thresh=(0, 50),
                                    s_thresh=(90, 255),
                                    v_thresh=(0, 255))
    # Extract white binary
    white_binary = color_threshold(image,
                                   h_thresh=(0, 255),
                                   s_thresh=(0, 30),
                                   v_thresh=(200, 255))
    # Combine color binaries
    color_binary = cv2.bitwise_or(yellow_binary, white_binary)

    # Convert imageorted image to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    # Apply gradient pipe line to L and S channel
    gradient_combined = gradient_pipe_line(l) + gradient_pipe_line(s)
    # Apply Gaussian blur
    gradient_combined_blur = cv2.GaussianBlur(gradient_combined, (5, 5), 0)
    # Gradient Binary
    gradient_binary = numpy.zeros_like(gradient_combined_blur)
    gradient_binary[gradient_combined_blur > 0.5] = 1

    # combine color and gradient filter
    lane_combined = cv2.bitwise_or(color_binary, gradient_binary)
    return lane_combined


def color_threshold(img, h_thresh=(0, 255), s_thresh=(0, 255), v_thresh=(0, 255)):
    img = numpy.copy(img)
    # Convert to HSV color space and separate channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(numpy.float)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Threshold color channel
    color_binary = numpy.zeros_like(s_channel)
    color_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) &
                 ((h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])) &
                 ((v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]))] = 1
    return color_binary


def gradient_pipe_line(image):
    img_g_mag = mag_thresh(image, 3, (20, 150))
    img_d_mag = dir_threshold(image, 3, (.6, 1.1))
    img_abs_x = abs_sobel_thresh(image, 'x', 5, (50, 200))
    img_abs_y = abs_sobel_thresh(image, 'y', 5, (50, 200))
    sobel_combined = numpy.zeros_like(img_d_mag)
    sobel_combined[((img_abs_x == 1) & (img_abs_y == 1)) | \
                   ((img_g_mag == 1) & (img_d_mag == 1))] = 1
    return sobel_combined


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = numpy.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = numpy.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(numpy.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = numpy.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, numpy.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = numpy.arctan2(numpy.absolute(sobely), numpy.absolute(sobelx))
    dir_binary = numpy.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    img = numpy.copy(img)
    if orient == 'x':
        # Sobel x
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # Take the derivative in x
    elif orient == 'y':
        # Sobel y
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # Take the derivative in y
    else:
        raise NameError('Please specify gradient orientation, x or y')
    # Absolute derivative to accentuate lines away from horizontal
    abs_sobel = numpy.absolute(sobel)
    scaled_sobel = numpy.uint8(255 * abs_sobel / numpy.max(abs_sobel))
    # Threshold gradient
    grad_binary = numpy.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary
