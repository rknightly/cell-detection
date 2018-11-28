from image import Image
import numpy as np


class Slide(Image):
    '''
    An image of a slide from a microscope. This slide is expected to contain
    at least some cells and not simply be noise or blank.
    '''
    @classmethod
    def from_cells(cls, cells, width, height):
        '''
        Creates and returns a slide of a given size where the
        only filled-in pixels are those that are part of the given cells
        :param cells: a list of cells
        :param width: the total width of the desired output image
        :param height: the total height of the desired output image
        :returns an image object with shape (width, height) and only the pixels
         contained in cells filled in
        '''
        pixel_values = np.zeros(shape=(height, width), dtype=int)
        image = cls(pixel_values)

        for cell in cells:
            for pixel_location in cell.pixel_locations:
                image.set_pixel(pixel_location[0], pixel_location[1], 1)

        return image

    def highlight_cells(self, cells):
        '''
        Add highlights to the image to show the discovered cells
        :returns None
        '''
        for cell in cells:
            for pixel_location in cell.pixel_locations:
                self.highlight_pixel(pixel_location[0], pixel_location[1])
