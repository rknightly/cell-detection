import imageio
import numpy as np
from matplotlib import pyplot as plt

class Image(object):
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values
        self.height = self.pixel_values.shape[0]
        self.width = self.pixel_values.shape[1]

    @classmethod
    def from_file_name(cls, file_name):
        pixel_values = imageio.imread(file_name)
        return cls(pixel_values)

    def get_pixel(self, x, y):
    	return self.pixel_values[y][x]

    def set_pixel(self, x, y, value):
    	self.pixel_values[y][x] = value

    def convert_to_grayscale(self):
        '''
        Converts the image to grayscale
        '''
        self.pixel_values = np.mean(self.pixel_values, axis=2)

    def normalize_pixels(self):
        '''
        Normalizes all of the values in the matrix to be within the range [0, 1]
        '''
        min = np.min(self.pixel_values)
        max = np.max(self.pixel_values)

        for y in range(self.height):
            for x in range(self.width):
            	# normalize to range [0,1]
                normalized_value = (self.get_pixel(x, y) - min) / (max-min)
                self.set_pixel(x, y, normalized_value)

    def apply_noise_gate(self, threshold):
        '''
        Applies a noise gate to a 2D matrix of an image.
        Any pixels below a certain threshold will be set to 0,
        any others will be set to 1
        '''
        mean_filled = np.sum(self.pixel_values) / np.count_nonzero(self.pixel_values)

        unfilled_count = 0
        filled_count = 0

        for x in range(self.width):
            for y in range(self.height):
                if self.get_pixel(x, y) < threshold:
                    self.set_pixel(x, y, 0)
                    unfilled_count += 1
                else:
                    self.set_pixel(x, y, 1)
                    filled_count += 1

    def expand_pixels(self):
        '''
        Merges the cells in a given image by simply expanding each filled pixel into its 4 neighboring pixels
        :param image: an image matrix of the image to be processed
        :returns an image matrix of the same shape as the input image after each pixel was expanded
        '''
        resulting_image = self.copy()
        for x in range(resulting_image.width):
            for y in range(resulting_image.height):
                if self.get_pixel(x, y) == 1:
                    if x > 0:
                        resulting_image.set_pixel(x-1, y, 1)
                    if x < self.width - 1:
                        resulting_image.set_pixel(x+1, y, 1)
                    if y > 0:
                        resulting_image.set_pixel(x, y-1, 1)
                    if y < self.height - 1:
                        resulting_image.set_pixel(x, y+1, 1)

        self.pixel_values = resulting_image.pixel_values

    def highlight_pixel(self, x, y):
    	# Each pixel will have [Red, Green, Blue]
        # Set green and blue channels to highlight the pixel
        pixel_value = self.get_pixel(x, y)
        pixel_value[1] = 180 # green channel
        pixel_value[2] = 180 # blue channel

    def add_to_plot(self, figure_num):
        plt.figure(figure_num)
        # Grayscale only has 1 color channel. The shape of a grayscale image will be (Y, X) vs. (Y, X, n) for a colored image
        is_grayscale = len(self.pixel_values.shape) == 2

        if is_grayscale:
            plt.imshow(self.pixel_values, cmap='gray', interpolation='nearest')
        else:
            plt.imshow(self.pixel_values, interpolation='nearest')

    def show(self):
        self.add_to_plot(0)
        plt.show()

    def copy(self):
    	return Image(self.pixel_values.copy())
