import numpy as np

from constants import SMALL_CELL_THRESHOLD
from constants import ORIGINAL_IMAGE_WIDTH


class Cell:
    '''
    Represents a cell visually present in a slide.
    '''
    def __init__(self, pixel_locations):
        '''
        Initializes a new Cell object
        :param pixel_locations: a list of all the (X, Y) pixel coordinates that
         are part of the cell in the slide image
        :returns None
        '''
        self.pixel_locations = pixel_locations

    def get_center(self):
        '''
        Finds the location of the center of the cell in the image
        :returns the center of the cell as an [X, Y] value pair in pixels
        '''
        return np.mean(self.pixel_locations, axis=0)

    def is_valid_size(self, image_width):
        '''
        Determines whether the cell is of a valid size to be a cell, otherwise
        it is deemed to be too small and likely noise
        :returns a boolean that is True if the cell is of a valid size
        '''
        image_ratio = image_width / ORIGINAL_IMAGE_WIDTH
        effective_threshold = SMALL_CELL_THRESHOLD * (image_ratio ** 2)

        return self.get_pixel_count() > effective_threshold

    def get_pixel_count(self):
        '''
        Gets the number of pixels that compose the cell in the slide
        :returns and integer number representing the number of pixels that
         represent the cell
        '''
        return len(self.pixel_locations)

    def serialized(self):
        '''
        Gives the current cell as a dictionary of values able to be written
        directly to a JSON file
        :returns a dictionary representing the cell
        '''
        return [
            {'x': pixel_location[0], 'y': pixel_location[1]}
            for pixel_location in self.pixel_locations
        ]
