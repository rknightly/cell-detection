from matplotlib import pyplot as plt
import json

from constants import FILTER_THRESHOLD
from slide import Slide
from cell import Cell


class CellDetector:
    '''
    Provides tools to detect the locations of cells in a microscope slide
    '''

    def __init__(self, input_file, output_file='', show_process=False):
        '''
        Initializes a cell detector to process the image at the given file
        :param file_name: a string of the name of the file that contains a
         slide with cells to process
        :returns None
        '''
        self.slide = Slide.from_file_name(input_file)
        self.filtered_slide = self.slide.copy()
        self.input_file = input_file
        self.output_file = output_file
        self.show_process = show_process

        self.figure_num = 0
        self.record_process(filtered=False)

    def detect_cells(self):
        '''
        Detects the cells in an image after filtering and processing it
        :returns a list of detected cells, where each cell is list of (X, Y)
         pixel coordinates that contain the cell
        '''
        self.filtered_slide.convert_to_grayscale()
        self.record_process()

        self.filtered_slide.normalize_pixels()
        self.filtered_slide.apply_noise_gate(FILTER_THRESHOLD)
        self.record_process()

        self.filtered_slide = self.remove_small_cells_from(self.filtered_slide)
        self.record_process()

        # Avoid unintentionally 'splitting' a single cell into multiple
        self.filtered_slide.expand_pixels()
        self.record_process()

        # Use this image to find the final estimates of cell locations
        cells_found = self.find_cells_raw(self.filtered_slide)

        self.slide.highlight_cells(cells_found)
        self.record_process(filtered=False)

        return cells_found

    def remove_small_cells_from(self, slide):
        possible_cells = self.find_cells_raw(slide)

        # Construct a new image that only contains the pixels that are part
        # of a reasonable sized "cell," which should
        # Exclude small areas of high intensity noise
        return Slide.from_cells(
            possible_cells,
            slide.width,
            slide.height)

    def find_cells_raw(self, slide):
        '''
        Searches a slide for all cells it contains which are not unusually
        small. (Without any pre-processing)
        :param slide: a grayscale slide with minimal noise if any
        :returns a list of detected cells
        '''
        cells_found = []
        working_slide = slide.copy()

        for x in range(slide.width):
            for y in range(slide.height):
                if working_slide.get_pixel(x, y) > 0:
                    cell, working_slide = self.explore_cell(
                        working_slide, x, y)
                    # ignore cells that are too small
                    if cell.is_valid_size(self.slide.width):
                        cells_found.append(cell)

        return cells_found

    def explore_cell(self, slide, base_x, base_y):
        '''
        Finds all pixels that are part of the cell that contains the base pixel
        :param slide: the slide to search
        :param base_x: the X coordinate of any pixel contained in the cell
        :param base_y: the Y coordinate of any pixel contained in the cell
        :returns cell: the entire cell that was found
        :returns slide: the slide with the current cell zeroed out
        '''
        cell_pixels = []

        def flood_fill(x, y):
            if slide.get_pixel(x, y) > 0:
                cell_pixels.append((x, y))
                slide.set_pixel(x, y, 0)
                # Invoke flood fill on all surrounding cells:
                if x > 0:
                    flood_fill(x - 1, y)
                if x < slide.width - 1:
                    flood_fill(x + 1, y)
                if y > 0:
                    flood_fill(x, y - 1)
                if y < slide.height - 1:
                    flood_fill(x, y + 1)

        flood_fill(base_x, base_y)

        return Cell(cell_pixels), slide

    @staticmethod
    def print_cell_results(cells):
        '''
        Prints the cells found to the console
        :param cells: a list of detected cells
        :returns None
        '''
        for cell in cells:
            center = cell.get_center()
            message = 'Found cell with {} pixels. '.format(
                cell.get_pixel_count())
            message += 'With center: {:.1f} {:.1f}'.format(
                center[0],
                center[1])

            print(message)

        print(f'Found {len(cells)} cells.')

    def write_cell_results(self, cells):
        '''
        Writes the cell results to a JSON file
        :param cells: a list of detected cells
        :returns None
        '''
        cells = [cell.serialized() for cell in cells]
        data = {'cells': cells}

        with open(self.output_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def run(self):
        '''
        Detects the cells in a particular image and writes the output
        :returns None
        '''
        cells = self.detect_cells()
        self.print_cell_results(cells)
        self.write_cell_results(cells)

        if self.show_process:
            self.show()

    def record_process(self, filtered=True):
        '''
        Add figures of the working slide to the plot to record the filtering
        process
        :param filtered: boolean of whether to record the filtered image rather
         than the raw slide
        :returns None
        '''
        if self.show_process:
            if filtered:
                self.filtered_slide.add_to_plot(self.figure_num)
            else:
                self.slide.add_to_plot(self.figure_num)
            self.figure_num += 1

    def show(self):
        '''
        Shows the plot containing the slide images
        '''
        plt.show()
