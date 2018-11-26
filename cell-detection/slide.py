from image import Image
import numpy as np

class Slide(Image):
    @classmethod
    def from_cells(cls, cells, width, height):
        '''
        Creates an image where the only filled in pixels are those that are part of the given cells
        :param image_shape: a tuple describing the shape of the image array
        :returns: a matrix representing the image where the only colored pixels are those contained in the given cells
        '''
        pixel_values = np.zeros(shape=(height, width))
        image = cls(pixel_values)

        for cell in cells:
            for pixel_location in cell:
                image.set_pixel(pixel_location[0], pixel_location[1], 1)

        return image

    def highlight_cells(self, cells):
        '''
        Add highlights to the image to show the discovered cells
        '''
        overlayed_cells = []
        # RGB

        for cell in cells:
            for pixel_location in cell:
                self.highlight_pixel(pixel_location[0], pixel_location[1])
