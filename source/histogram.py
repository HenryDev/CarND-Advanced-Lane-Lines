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
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = current_left - margin
        win_xleft_high = current_left + margin
        win_xright_low = current_right - margin
        win_xright_high = current_right + margin
        # Draw the windows on the visualization image
        cv2.rectangle(output_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
            nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
            nonzero_x < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_index.append(good_left_inds)
        right_index.append(good_right_inds)
        # If you found > minimum_pixels pixels, recenter next window on their mean position
        if len(good_left_inds) > minimum_pixels:
            current_left = numpy.int(numpy.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minimum_pixels:
            current_right = numpy.int(numpy.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_index = numpy.concatenate(left_index)
    right_index = numpy.concatenate(right_index)

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_index]
    lefty = nonzero_y[left_index]
    rightx = nonzero_x[right_index]
    righty = nonzero_y[right_index]
    return leftx, lefty, rightx, righty


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
