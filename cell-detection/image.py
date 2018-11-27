import imageio
import numpy as np
from matplotlib import pyplot as plt


class Image(object):
    '''
    A 2D image, represented as an array of pixels.
    The origin of the image is at the top-left,
    and X and Y coordinates of the pixels are positive.
    The image can have dimensions of any (height, width, color_channels)
    '''

    def __init__(self, pixel_values):
        '''
        Initializes a new image given it's pixel values
        :param pixel_values: a numpy array of pixel values of
         the image, of the shape shape (height, width, color_channels)
        :returns None
        '''
        self.pixel_values = pixel_values
        self.height = self.pixel_values.shape[0]
        self.width = self.pixel_values.shape[1]

    @classmethod
    def from_file_name(cls, file_name):
        '''
        Creates an Image object from an image file
        :param file_name: a string of the image file name, including
         the path and extension
        :returns the created image object
        '''
        pixel_values = imageio.imread(file_name)
        return cls(pixel_values)

    def convert_to_grayscale(self):
        '''
        Converts the image to grayscale
        :returns None
        '''
        self.pixel_values = np.mean(self.pixel_values, axis=2)

    def normalize_pixels(self):
        '''
        Normalizes all of the pixel values to be within the range [0, 1]
        :returns None
        '''
        min = np.min(self.pixel_values)
        max = np.max(self.pixel_values)

        self.pixel_values = (self.pixel_values - min) / (max - min)

    def apply_noise_gate(self, threshold):
        '''
        Applies a noise gate to the image.
        Any pixels below the given threshold will be set to 0,
        any others will be set to 1.
        :returns None
        '''
        for x in range(self.width):
            for y in range(self.height):
                if self.get_pixel(x, y) < threshold:
                    self.set_pixel(x, y, 0)
                else:
                    self.set_pixel(x, y, 1)
        # The pixels are now int, not floats
        self.pixel_values = self.pixel_values.astype(int)

    def expand_pixels(self):
        '''
        Expands the filled pixels in an image by filling the four surrounding
        pixels of any already filled pixel. Results in a 'blurring' or
        'pixel expanding' effect
        :returns None
        '''
        resulting_image = self.copy()
        for x in range(resulting_image.width):
            for y in range(resulting_image.height):
                if self.get_pixel(x, y) == 1:
                    if x > 0:
                        resulting_image.set_pixel(x - 1, y, 1)
                    if x < self.width - 1:
                        resulting_image.set_pixel(x + 1, y, 1)
                    if y > 0:
                        resulting_image.set_pixel(x, y - 1, 1)
                    if y < self.height - 1:
                        resulting_image.set_pixel(x, y + 1, 1)

        self.pixel_values = resulting_image.pixel_values

    def highlight_pixel(self, x, y):
        '''
        Add a light teal-colored highlight at the given pixel location
        :param x: the x position of the pixel to fill (origin at top-left)
        :param y: the y position of the pixel to fill (origin at top-left)
        :returns None
        '''
        # Each pixel will have [Red, Green, Blue]
        # Set green and blue channels to highlight the pixel
        pixel_value = self.get_pixel(x, y)
        pixel_value[1] = 180  # green channel
        pixel_value[2] = 180  # blue channel

    def add_to_plot(self, figure_num):
        '''
        Add this image to the pyplot plot in the specified figure
        :param figure_num: the number of the figure to contain the image
        :returns None
        '''
        plt.figure(figure_num)
        # Grayscale only has 1 color channel, so a shape of (height, width)
        # RGB has color channels, so a shape of (height, width, channels)
        is_grayscale = len(self.pixel_values.shape) == 2

        if is_grayscale:
            plt.imshow(self.pixel_values, cmap='gray', interpolation='nearest')
        else:
            plt.imshow(self.pixel_values, interpolation='nearest')

    def get_pixel(self, x, y):
        '''
        Gets the value of a pixel at a given location, with origin at the
        upper left and (x, y) coordinates are positive
        :param x: the x value of the pixel location (origin at top-left)
        :param y: the y value of the pixel location (origin at top-left)
        :returns the value of the pixel located at the specified coordinates
        '''
        # [row][column] -> [y][x]  (2D array indices)
        return self.pixel_values[y][x]

    def set_pixel(self, x, y, value):
        '''
        Sets the value of a pixel at a given location, with origin at the
        upper left and x, y coordinates are positive
        :param x: the x value of the pixel location (origin at top-left)
        :param y: the y value of the pixel location (origin at top-left)
        :param value: the value to set the specified pixel to
        '''
        # [row][column] -> [y][x]  (2D array indices)
        self.pixel_values[y][x] = value

    def copy(self):
        '''
        Creates a new, independent copy of this image
        :returns the copy of this image object
        '''
        return Image(self.pixel_values.copy())
