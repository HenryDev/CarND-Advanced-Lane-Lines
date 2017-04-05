class Tracker:
    def __init__(self, box_width, box_height, margin, vertical_meters_per_pixel=1, horizontal_meters_per_pixel=1,
                 smooth_factor=15):
        self.recent_centers = []
        self.box_width = box_width
        self.box_height = box_height
        self.margin = margin
        self.vertical_meters_per_pixel = vertical_meters_per_pixel
        self.horizontal_meters_per_pixel = horizontal_meters_per_pixel
        self.smooth_factor = smooth_factor
