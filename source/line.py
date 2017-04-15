import collections
import numpy


class Line:
    def __init__(self):
        self.detected = False
        self.recent_fits = collections.deque(maxlen=10)
        self.best_fit = None
        self.current_fit = [numpy.array([False])]

    def update_fit(self, line_fit):
        self.current_fit = line_fit
        self.recent_fits.append(line_fit)
        self.best_fit = numpy.average(self.recent_fits, axis=0)
