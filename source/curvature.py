import cv2
import numpy


def draw_curve(image, warped, window_width, window_height, left_x, right_x):
    y_values = range(0, warped.shape[0])
    res_yvals = numpy.arange(warped.shape[0] - window_height / 2, 0, -window_height)

    left_fit = numpy.polyfit(res_yvals, left_x, 2)
    left_fit_x = left_fit[0] * y_values * y_values + left_fit[1] * y_values + left_fit[2]
    left_fit_x = numpy.array(left_fit_x, numpy.int32)

    right_fit = numpy.polyfit(res_yvals, right_x, 2)
    right_fit_x = right_fit[0] * y_values * y_values + right_fit[1] * y_values + right_fit[2]
    right_fit_x = numpy.array(right_fit_x, numpy.int32)

    left_lane = numpy.array(list(
        zip(numpy.concatenate((left_fit_x - window_width / 2, left_fit_x[::-1] + window_width / 2), axis=0),
            numpy.concatenate((y_values, y_values[::-1]), axis=0))), numpy.int32)
    right_lane = numpy.array(list(
        zip(numpy.concatenate((right_fit_x - window_width / 2, right_fit_x[::-1] + window_width / 2), axis=0),
            numpy.concatenate((y_values, y_values[::-1]), axis=0))), numpy.int32)

    road = numpy.zeros_like(image)

    cv2.fillPoly(road, [left_lane], [255, 0, 0])
    cv2.fillPoly(road, [right_lane], [0, 0, 255])
    return road
