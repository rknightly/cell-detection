from slide import Slide
import numpy as np
from constants import FILTER_THRESHOLD
from constants import SMALL_CELL_THRESHOLD
from constants import ORIGINAL_IMAGE_WIDTH

class CellDetector:
    '''
    Provides the mechanisms to detect the locations of cells in a slide from a microscope
    '''
    def __init__(self, file_name):
        '''
        Initializes a cell detector to process the image at the given file
        :param file_name: a string of the name of the file that contains a slide with cells
         to process
        :returns None
        '''
        self.slide = Slide.from_file_name(file_name)
        self.filtered_slide = self.slide.copy()

    def detect_cells(self):
        '''
        Detects the cells in an image after filtering and processing it
        :returns a list of detected cells, where each cell is list of (X, Y) pixel coordinates that contain the cell
        '''
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
        self.slide.add_to_plot(5)

        return cells_found

    def find_cells_raw(self, slide):
        '''
        Without any pre-processing, this searches a slide for all cells it contains which are not unusually small.
        :param slide: a grayscale slide with minimal noise if any
        :returns a list of detected cells, where each cell is list of (X, Y) pixel coordinates that contain the cell
        '''

        image_ratio = slide.width / ORIGINAL_IMAGE_WIDTH
        effective_threshold = SMALL_CELL_THRESHOLD * (image_ratio ** 2)
        cells_found = []
        working_slide = slide.copy()

        for x in range(slide.width):
            for y in range(slide.height):
                if working_slide.get_pixel(x, y) > 0:
                    cell_pixels, working_slide = self.explore_cell(working_slide, x, y)
                    if len(cell_pixels) > effective_threshold: # ignore cells that are too small
                        cells_found.append(cell_pixels)

        return cells_found

    def explore_cell(self, slide, base_x, base_y):
        '''
        Finds all pixels that are part of the cell that contains the base pixel
        :param slide: the slide to search
        :param base_x: the X coordinate of any pixel contained in the cell
        :param base_y: the Y coordinate of any pixel contained in the cell
        :returns cell_pixels: a list of all the (X, Y) pixel coordinates that are part of the cell
        :returns slide: the given slide with all the pixels of the current cell changed to 0
        '''
        cell_pixels = []

        def flood_fill(x, y):
            if slide.get_pixel(x, y) > 0:
                cell_pixels.append( (x, y) )
                slide.set_pixel(x, y, 0)
                # Invoke flood fill on all surrounding cells:
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
        :param cells: :returns a list of detected cells, where each cell is list of (X, Y) pixel coordinates that contain the cell
        :returns None
        '''
        for cell_pixels in cells:
            center = np.mean(cell_pixels, axis=0)
            print(f'Found cell with {len(cell_pixels)} pixels. With center: {center[0]:.1f}, {center[1]:.1f}')

        print(f'Found {len(cells)} cells.')

    def run(self):
        '''
        Runs the cell detector, by detecting the cells in a particular image and writing the output
        :returns None
        '''
        cells = self.detect_cells()
        self.print_cell_results(cells)

    def show(self):
        '''
        Shows the plot containing the slide images
        '''
        plt.show()
