import numpy


class Line:
    def __init__(self):
        self.was_line_detected = False
        self.last_x_fit = []
        self.average_x = None
        self.average_polynomial_coefficient = None
        self.current_fit = [numpy.array([False])]
        self.radius_of_curvature = None
        self.position_error = None
        self.coefficient_difference = numpy.array([0, 0, 0], dtype='float')
        self.x_all = None
        self.y_all = None
