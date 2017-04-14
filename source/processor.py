import pickle
import cv2

import numpy

from source.curvature import draw_curve, add_text, find_center_diff, find_radius, overlay_curvature_pos, \
    overlay_lane_detection, calc_offset, calc_radius, polyfit_pixels
from source.histogram import extend_fit, sliding_windows
from source.line import Line
from source.thresholds import threshold_pipeline
from source.tracker import Tracker, draw_windows
from source.transformer import transform


def process_image(image):
    distortion_pickle = pickle.load(open('../calibration_pickle.p', 'rb'))
    camera_matrix = distortion_pickle['cameraMatrix']
    dist_coeffs = distortion_pickle['distCoeffs']

    image = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)

    processed_image = threshold_pipeline(image)

    warped, m_inverse = transform(image, processed_image)

    left_line = Line()
    right_line = Line()
    if left_line.detected:
        leftx, lefty, rightx, righty = extend_fit(warped, left_line.current_fit, right_line.current_fit)

    else:
        leftx, lefty, rightx, righty = sliding_windows(warped)

    left_fit, right_fit = polyfit_pixels(leftx, lefty, rightx, righty)

    left_line.update_fit(left_fit)

    offset = calc_offset(warped, left_fit, right_fit)

    overlay = overlay_lane_detection(image, warped, m_inverse, left_line.best_fit, right_line.best_fit)

    left_curverad, right_curverad = calc_radius(warped, leftx, lefty, rightx, righty)
    weighted_road = overlay_curvature_pos(overlay, left_curverad, right_curverad, offset)

    return weighted_road
