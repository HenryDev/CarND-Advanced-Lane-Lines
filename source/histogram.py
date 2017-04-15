import cv2
import numpy


def sliding_windows(warped):
    histogram = numpy.sum(warped[int(warped.shape[0] / 2):, :], axis=0)
    output_image = numpy.dstack((warped, warped, warped)) * 255
    midpoint = numpy.int(histogram.shape[0] / 2)
    left_peak = numpy.argmax(histogram[:midpoint])
    right_peak = numpy.argmax(histogram[midpoint:]) + midpoint

    number_of_windows = 9
    window_height = numpy.int(warped.shape[0] / number_of_windows)
    nonzero = warped.nonzero()
    nonzero_y = numpy.array(nonzero[0])
    nonzero_x = numpy.array(nonzero[1])
    current_left = left_peak
    current_right = right_peak
    margin = 100
    minimum_pixels = 50
    left_index = []
    right_index = []

    for window in range(number_of_windows):
        y_low = warped.shape[0] - (window + 1) * window_height
        y_high = warped.shape[0] - window * window_height
        x_left_low = current_left - margin
        x_left_high = current_left + margin
        x_right_low = current_right - margin
        x_right_high = current_right + margin
        cv2.rectangle(output_image, (x_left_low, y_low), (x_left_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(output_image, (x_right_low, y_low), (x_right_high, y_high), (0, 255, 0), 2)
        left_hit = ((nonzero_y >= y_low)
                    & (nonzero_y < y_high)
                    & (nonzero_x >= x_left_low)
                    & (nonzero_x < x_left_high)).nonzero()[0]
        right_hit = ((nonzero_y >= y_low)
                     & (nonzero_y < y_high)
                     & (nonzero_x >= x_right_low)
                     & (nonzero_x < x_right_high)).nonzero()[0]
        left_index.append(left_hit)
        right_index.append(right_hit)
        if len(left_hit) > minimum_pixels:
            current_left = numpy.int(numpy.mean(nonzero_x[left_hit]))
        if len(right_hit) > minimum_pixels:
            current_right = numpy.int(numpy.mean(nonzero_x[right_hit]))

    left_index = numpy.concatenate(left_index)
    right_index = numpy.concatenate(right_index)

    left_x = nonzero_x[left_index]
    left_y = nonzero_y[left_index]
    right_x = nonzero_x[right_index]
    right_y = nonzero_y[right_index]
    return left_x, left_y, right_x, right_y


def extend_fit(warped, left_fit, right_fit):
    nonzero = warped.nonzero()
    nonzero_y = numpy.array(nonzero[0])
    nonzero_x = numpy.array(nonzero[1])
    margin = 100
    left_index = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - margin))
                  & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + margin)))
    right_index = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - margin))
                   & (nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + margin)))
    left_x = nonzero_x[left_index]
    left_y = nonzero_y[left_index]
    right_x = nonzero_x[right_index]
    right_y = nonzero_y[right_index]
    return left_x, left_y, right_x, right_y
