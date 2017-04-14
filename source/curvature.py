import cv2
import numpy as np, numpy


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

    curve_radius, weighted_road = find_radius(curve_centers, image, left_x, m_inverse, res_yvals, road, road_background,
                                              y_values)

    center_diff = find_center_diff(curve_centers, left_fit_x, right_fit_x, warped)

    weighted_road = add_text(center_diff, curve_radius, weighted_road)

    return weighted_road


def find_radius(curve_centers, image, left_x, m_inverse, res_yvals, road, road_background, y_values):
    image_size = (image.shape[1], image.shape[0])
    warped_road = cv2.warpPerspective(road, m_inverse, image_size, flags=cv2.INTER_LINEAR)
    warped_road_background = cv2.warpPerspective(road_background, m_inverse, image_size, flags=cv2.INTER_LINEAR)
    base = cv2.addWeighted(image, 1, warped_road_background, -1, 0)
    weighted_road = cv2.addWeighted(base, 1, warped_road, 0.7, 0)
    curve_fit_cr = numpy.polyfit(numpy.array(res_yvals, numpy.float32) * curve_centers.vertical_meters_per_pixel,
                                 numpy.array(left_x, numpy.float32) * curve_centers.horizontal_meters_per_pixel, 2)
    curve_radius = ((1 + (2 * curve_fit_cr[0] * y_values[-1] * curve_centers.vertical_meters_per_pixel + curve_fit_cr[
        1]) ** 2) ** 1.5) / numpy.absolute(2 * curve_fit_cr[0])
    return curve_radius, weighted_road


def find_center_diff(curve_centers, left_fit_x, right_fit_x, warped):
    camera_center = (left_fit_x[-1] + right_fit_x[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * curve_centers.horizontal_meters_per_pixel
    return center_diff


def add_text(center_diff, curve_radius, weighted_road):
    side_position = 'left'
    if center_diff <= 0:
        side_position = 'right'
    cv2.putText(weighted_road, 'radius of curvature = ' + str(round(curve_radius, 3)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(weighted_road, 'car is ' + str(abs(round(center_diff, 3))) + 'm ' + side_position + ' of center',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return weighted_road


def overlay_curvature_pos(overlay, left_curverad, right_curverad, offset):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, "left line radius: {0:.5g} m".format(left_curverad), (50, 50), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(overlay, "right line radius: {0:.5g} m".format(right_curverad), (50, 100), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    if offset > 0:
        rel_dir = "right"
    else:
        rel_dir = "left"

    cv2.putText(overlay, "Vehicle is {0:.2g}m {1} of center".format(np.absolute(offset), rel_dir), (50, 150), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    return overlay


def overlay_lane_detection(image, binary_warped, Minv, left_fit, right_fit):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(image).astype(np.uint8)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    overlay = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return overlay
def calc_offset(binary_warped, left_fit, right_fit):
    y_eval = binary_warped.shape[0] - 1
    xm_per_pix = 3.7/920 # meters per pixel in x dimension

    bottom_left_x = np.polyval(left_fit, y_eval)
    bottom_right_x = np.polyval(right_fit, y_eval)
    offset = (binary_warped.shape[1]/2 - (bottom_left_x + bottom_right_x)/2) * xm_per_pix
    return offset
def calc_radius(binary_warped, leftx, lefty, rightx, righty):
    y_eval = binary_warped.shape[0] - 1
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 15/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/920 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radius of curvature in meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad
def polyfit_pixels(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit