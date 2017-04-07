import cv2
import numpy


def draw_curve(warped, image, window_height, window_width, left_x, right_x, m_inverse):
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
    road_background = numpy.zeros_like(image)

    cv2.fillPoly(road, [left_lane], [255, 0, 0])
    cv2.fillPoly(road, [right_lane], [0, 0, 255])
    cv2.fillPoly(road_background, [left_lane], [255, 255, 255])
    cv2.fillPoly(road_background, [right_lane], [255, 255, 255])

    image_size = (image.shape[1], image.shape[0])
    warped_road = cv2.warpPerspective(road, m_inverse, image_size, flags=cv2.INTER_LINEAR)
    warped_road_background = cv2.warpPerspective(road_background, m_inverse, image_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(image, 1, warped_road_background, -1, 0)
    weighted_road = cv2.addWeighted(base, 1, warped_road, 1, 0)

    return weighted_road
