from slide import Slide
import numpy as np
from constants import FILTER_THRESHOLD
from constants import SMALL_CELL_THRESHOLD
from constants import ORIGINAL_IMAGE_WIDTH

class CellDetector:
    def __init__(self, file_name):
        self.slide = Slide.from_file_name(file_name)
        self.filtered_slide = self.slide.copy()

    def detect_cells(self):
        self.filtered_slide.convert_to_grayscale()
        self.filtered_slide.normalize_pixels()
        self.filtered_slide.add_to_plot(0)
        self.filtered_slide.apply_noise_gate(FILTER_THRESHOLD)
        self.filtered_slide.add_to_plot(1)

        # Find the possible cells, which are reasonably sized groups of colored pixels
        possible_cells = self.find_cells_raw(self.filtered_slide)

        # Construct a new image that only contains the pixels that are part of a reasonable sized "cell," which should
        # Exclude small areas of high intensity noise
        self.filtered_slide = Slide.from_cells(possible_cells, self.filtered_slide.width, self.filtered_slide.height)
        self.filtered_slide.add_to_plot(3)
        # Try to combine very close "cells" so no single cell is split up and percieved as multiple cells
        self.filtered_slide.expand_pixels()
        self.filtered_slide.add_to_plot(4)


        # Use this image to find the final estimates of cell locations
        cells_found = self.find_cells_raw(self.filtered_slide)

        self.slide.highlight_cells(cells_found)
        self.slide.show()


        return cells_found

    def find_cells_raw(self, slide):
        '''
        Searches an image matrix for all cells it contains which are not unusually small.
        Prints the output to the screen
        :param image: a 2D matrix of the grayscale image
        :returns a list of the discovered cells where each cell is a list of pixels that
        represent the cell
        '''

        image_ratio = slide.width / ORIGINAL_IMAGE_WIDTH
        effective_threshold = SMALL_CELL_THRESHOLD * (image_ratio ** 2)
        cells_found = []
        working_slide = slide.copy()

        for x in range(slide.width):
            for y in range(slide.height):
                if working_slide.get_pixel(x, y) > 0.0:
                    cell_pixels, working_slide = self.explore_cell(working_slide, x, y)
                    if len(cell_pixels) > effective_threshold: # ignore cells that are too small
                        cells_found.append(cell_pixels)

        return cells_found

    def explore_cell(self, slide, base_x, base_y):
        '''
        Finds all pixels that are part of the cell that contains the
        pixel at the location (initial_x, initial_y)
        :param matrix: a 2D array of the pixels of the image
        :param initial_x: the X coordinate of the starting pixel of the cell
        :param initial_y: the Y coordinate of the starting pixel of the cell
        :returns cell_pixels: a list of all the pixels that were part of the cell
        :returns matrix: the image matrix after the pixels of the current cell were changed to 0
        '''
        cell_pixels = []

        def flood_fill(x, y):
            # stop clause - not reinvoking for 0, only for >0
            if slide.get_pixel(x, y) > 0.0:
                cell_pixels.append( (x, y) )
                slide.set_pixel(x, y, 0)
                # recursively invoke flood fill on all surrounding cells:
                if x > 0:
                    flood_fill(x-1,y)
                if x < slide.width - 1:
                    flood_fill(x+1,y)
                if y > 0:
                    flood_fill(x,y-1)
                if y < slide.height - 1:
                    flood_fill(x,y+1)

        flood_fill(base_x, base_y)

        return cell_pixels, slide

    @staticmethod
    def print_cell_results(cells):
        '''
        Prints the cells found to the console
        :param cells: a list of cells where each cell is a list of pixel locations that contain the cell
        :returns None
        '''
        for cell_pixels in cells:
            center = np.mean(cell_pixels, axis=0)
            print(f'Found cell with {len(cell_pixels)} pixels. With center: {center[0]:.1f}, {center[1]:.1f}')

        print(f'Found {len(cells)} cells.')

    def run(self):
        cells = self.detect_cells()
        self.print_cell_results(cells)
