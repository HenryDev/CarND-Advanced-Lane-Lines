import cv2
import numpy

xm_per_pix = 3.7 / 700
ym_per_pix = 30 / 720


def add_text(overlay, radius, offset):
    cv2.putText(overlay, "radius: {0:.5g} m".format(radius), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if offset > 0:
        side = "right"
    else:
        side = "left"
    cv2.putText(overlay, "car is {0:.2g}m {1} of center".format(numpy.absolute(offset), side), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return overlay


def overlay_lane_detection(image, warped, m_inverse, left_fit, right_fit):
    color_warp = numpy.zeros_like(image).astype(numpy.uint8)

    plot_y = numpy.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fit = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    left_points = numpy.array([numpy.transpose(numpy.vstack([left_fit, plot_y]))])
    right_points = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right_fit, plot_y])))])
    points = numpy.hstack((left_points, right_points))

    cv2.fillPoly(color_warp, numpy.int_([points]), (0, 255, 0))

    new_warp = cv2.warpPerspective(color_warp, m_inverse, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 1, new_warp, 0.3, 0)
    return overlay


def calc_offset(warped, left_fit, right_fit):
    y_eval = warped.shape[0] - 1
    bottom_left_x = numpy.polyval(left_fit, y_eval)
    bottom_right_x = numpy.polyval(right_fit, y_eval)
    offset = (warped.shape[1] / 2 - (bottom_left_x + bottom_right_x) / 2) * xm_per_pix
    return offset


def calc_radius(warped, left_x, left_y, right_x, right_y):
    y_eval = warped.shape[0] - 1

    left_curve = numpy.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_curve = numpy.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
    left_radius = ((1 + (2 * left_curve[0] * y_eval * ym_per_pix + left_curve[1]) ** 2) ** 1.5) / numpy.absolute(
        2 * left_curve[0])
    right_radius = ((1 + (2 * right_curve[0] * y_eval * ym_per_pix + right_curve[1]) ** 2) ** 1.5) / numpy.absolute(
        2 * right_curve[0])

    return left_radius, right_radius


def polyfit_pixels(left_x, left_y, right_x, right_y):
    left_fit = numpy.polyfit(left_y, left_x, 2)
    right_fit = numpy.polyfit(right_y, right_x, 2)
    return left_fit, right_fit
