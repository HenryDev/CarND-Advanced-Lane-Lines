import cv2
import numpy


def draw_curve(warped, image, left_x, right_x, m_inverse, curve_centers):
    y_values = range(0, warped.shape[0])
    res_yvals = numpy.arange(warped.shape[0] - curve_centers.window_height / 2, 0, -curve_centers.window_height)

    left_fit = numpy.polyfit(res_yvals, left_x, 2)
    left_fit_x = left_fit[0] * y_values * y_values + left_fit[1] * y_values + left_fit[2]
    left_fit_x = numpy.array(left_fit_x, numpy.int32)

    right_fit = numpy.polyfit(res_yvals, right_x, 2)
    right_fit_x = right_fit[0] * y_values * y_values + right_fit[1] * y_values + right_fit[2]
    right_fit_x = numpy.array(right_fit_x, numpy.int32)

    left_lane = numpy.array(list(zip(numpy.concatenate(
        (left_fit_x - curve_centers.window_width / 2, left_fit_x[::-1] + curve_centers.window_width / 2), axis=0),
        numpy.concatenate((y_values, y_values[::-1]), axis=0))), numpy.int32)
    right_lane = numpy.array(list(zip(numpy.concatenate(
        (right_fit_x - curve_centers.window_width / 2, right_fit_x[::-1] + curve_centers.window_width / 2), axis=0),
        numpy.concatenate((y_values, y_values[::-1]), axis=0))), numpy.int32)
    inner_lane = numpy.array(list(zip(numpy.concatenate(
        (left_fit_x + curve_centers.window_width / 2, right_fit_x[::-1] - curve_centers.window_width / 2), axis=0),
        numpy.concatenate((y_values, y_values[::-1]), axis=0))), numpy.int32)

    road = numpy.zeros_like(image)
    road_background = numpy.zeros_like(image)

    cv2.fillPoly(road, [left_lane], [255, 0, 0])
    cv2.fillPoly(road, [right_lane], [0, 0, 255])
    cv2.fillPoly(road, [inner_lane], [0, 255, 0])
    cv2.fillPoly(road_background, [left_lane], [255, 255, 255])
    cv2.fillPoly(road_background, [right_lane], [255, 255, 255])

    image_size = (image.shape[1], image.shape[0])
    warped_road = cv2.warpPerspective(road, m_inverse, image_size, flags=cv2.INTER_LINEAR)
    warped_road_background = cv2.warpPerspective(road_background, m_inverse, image_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(image, 1, warped_road_background, -1, 0)
    weighted_road = cv2.addWeighted(base, 1, warped_road, 0.7, 0)

    curve_fit_cr = numpy.polyfit(numpy.array(res_yvals, numpy.float32) * curve_centers.vertical_meters_per_pixel,
                                 numpy.array(left_x, numpy.float32) * curve_centers.horizontal_meters_per_pixel, 2)
    curve_radius = ((1 + (2 * curve_fit_cr[0] * y_values[-1] * curve_centers.vertical_meters_per_pixel + curve_fit_cr[
        1]) ** 2) ** 1.5) / numpy.absolute(2 * curve_fit_cr[0])

    camera_center = (left_fit_x[-1] + right_fit_x[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * curve_centers.horizontal_meters_per_pixel
    side_position = 'left'
    if center_diff <= 0:
        side_position = 'right'
    cv2.putText(weighted_road, 'radius of curvature = ' + str(round(curve_radius, 3)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(weighted_road, 'car is ' + str(abs(round(center_diff, 3))) + 'm ' + side_position + ' of center',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return weighted_road
