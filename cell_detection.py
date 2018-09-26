import numpy as np
import imageio
from matplotlib import pyplot as plt


# The filter threshold for the normalized grayscale image.
# Pixels at or below this value are considered noise.
FILTER_THRESHOLD = 0.3
# The threshold number for what constitutes a cell that is unusually small. 
# Cells at or below this size are considered noise.
SMALL_CELL_THRESHOLD = 15


def main():
    '''
    Main function of the program.
    Reads the image from the file, displays it, filters it and displays it again.
    Finally, it detects the cells in the image and prints the locations
    '''
    # Read and process image
    original_image = read_image('data/raw/testSlide1.png')
    grayscale_image = convert_to_grayscale(np.copy(original_image)) # Make copy so we can still have older versions
    filtered_image = filter_noise(normalize(np.copy(grayscale_image)))

    # Find the possible cells that are groups of colored pixels
    possible_cells = detect_cells_in(np.copy(filtered_image))
    possible_cells = remove_small_cells_in(possible_cells)

    # Construct a new image without the pixels that were part of an oddly small "cell"
    # This can be viewed as a more sophisticated filter than previously, as these oddly
    # small pixel groups are probably just noise
    image_without_small_cells = construct_image_from_cells(grayscale_image, possible_cells)

    # Blur the image so that single cells aren't split up and percieved as multiple cells
    blurred_image = blur_image(image_without_small_cells)

    # Use this filtered, blurred image to find the final estimates of cell locations
    cells_found = detect_cells_in(np.copy(blurred_image))
    cells_found = remove_small_cells_in(cells_found)
    print_cell_results(cells_found)

    # Show the original image with the proposed cell locations highlighted on it
    cell_overlay = highlight_discovered_cells(np.copy(original_image), cells_found)

    # Show the image at different stages of processing
    display_images([original_image, grayscale_image, filtered_image, blurred_image, cell_overlay])


def read_image(file_name):
    '''
    Reads the image from the file and returns it as a matrix
    :param file_name: a str of the file name of the image
    :returns a numpy array of size (X, Y, n) where X and Y are the image dimensions
    and n is the number of color channels
    '''
    # Original image has R, G, B channels
    # Shape: (X pixels, Y pixels, 3 color channels)
    image = imageio.imread(file_name)

    return image


def convert_to_grayscale(image):
    '''
    Converts a given image to grayscale
    :param image: a matrix of dimension (X, Y, n) where X and Y are the width and height
    of the image and n color channels (likely 3 for RGB)
    :returns a matrix of dimension (X, Y, 1) of the original image in grayscale
    '''
    grayscale_image = np.mean(image, axis=2)
    print(f'grayscale shape: {grayscale_image.shape}')
    print(f'max grayscale pixel: {np.max(grayscale_image):0.3f}')

    return grayscale_image


def normalize(matrix):
    '''
    Normalizes all of the values in the matrix to be within the range [0, 1]
    :param matrix: a 2D array of numbers to be normalized
    :returns a numpy array of the same size with the numbers within range [0, 1]
    '''
    min = np.min(matrix)
    max = np.max(matrix)

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            matrix[x][y] = (matrix[x][y] - min)/(max-min) # normalize to range 0,1

    return matrix


def filter_noise(image):
    '''
    Applies a noise gate to a 2D matrix of an image.
    Any pixels below a certain threshold will be set to 0,
    any others will be set to 1
    :param image: a 2D array of the pixel values of the image
    :returns: a 2D array of the same size after the filter was applied
    '''
    print("Pre-processed mean: " + str(np.mean(image)))

    print(image.shape)
    mean_filled = np.sum(image) / np.count_nonzero(image)
    print("MEAN FILLED: " + str(mean_filled))

    unfilled_count = 0
    filled_count = 0

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # print(image[x][y])
            if image[x][y] < FILTER_THRESHOLD:
                image[x][y] = 0
                unfilled_count += 1
            else:
                # print("Filled pixels")
                image[x][y] = 1
                filled_count += 1

    print(f'Filled: {filled_count}, Unfilled: {unfilled_count}')
    print(np.mean(image))

    return image


def detect_cells_in(image):
    '''
    Searches an image matrix for all cells it contains.
    Prints the output to the screen
    :param image: a 2D matrix of the grayscale image
    :returns None
    '''
    cells_found = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] > 0.0:
                cell_pixels, image = explore_cell(image, x, y)
                cells_found.append(cell_pixels)

    return cells_found


def explore_cell(matrix, initial_x, initial_y):
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
        if matrix[x][y] > 0.0:
            cell_pixels.append(tuple([x, y]))
            matrix[x][y] = 0.0 
            #recursively invoke flood fill on all surrounding cells:
            if x > 0:
                flood_fill(x-1,y)
            if x < len(matrix[y]) - 1:
                flood_fill(x+1,y)
            if y > 0:
                flood_fill(x,y-1)
            if y < len(matrix) - 1:
                flood_fill(x,y+1)

    flood_fill(initial_x, initial_y)

    return cell_pixels, matrix


def remove_small_cells_in(cells):
    return [cell for cell in cells if len(cell) > SMALL_CELL_THRESHOLD]


def construct_image_from_cells(example_image, cells):
    image = np.zeros(shape=example_image.shape)
    for cell in cells:
        for pixel in cell:
            image[pixel[0]][pixel[1]] = 1

    return image


def blur_image(image):
    final_image = np.copy(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 1:
                if x > 0:
                    final_image[x-1][y] = 1
                if x < len(image[y]) - 1:
                    final_image[x+1][y] = 1
                if y > 0:
                    final_image[x][y-1] = 1
                if y < len(image) - 1:
                    final_image[x][y+1] = 1

    return final_image


def highlight_discovered_cells(image, cells_found):
    '''
    Add highlights to the original image to show the discovered cells
    :param image: the original RGB image to overlay
    :param cells_found: an array of cells, where each cell is an array of the pixels that represent that cell
    :returns an array representing an image with the overlays added
    '''
    overlayed_cells = []
    # RGB

    for cell in cells_found:
        for pixel in cell:
            # Each pixel will have [Red, Green, Blue]
            # Here we are changing green and blue to highlight the pixels where the cells were found
            image[pixel[0]][pixel[1]][1] = 180
            image[pixel[0]][pixel[1]][2] = 180

    return image


def print_cell_results(cells):
    for cell_pixels in cells:
        center = np.mean(cell_pixels, axis=0)
        print(f'Found cell with {len(cell_pixels)} pixels. With center: {center[0]:.1f}, {center[1]:.1f}')

    print(f'Found {len(cells)} cells.')


def display_images(images):
    for x, image in enumerate(images):
        plt.figure(x)
        # Grayscale only has 1 color channel. The shape of a grayscale image will be (X, Y) vs. (X, Y, n) for a colored image
        is_grayscale = len(image.shape) == 2    

        if is_grayscale:
            plt.imshow(image, cmap='gray', interpolation='nearest')
        else:
            plt.imshow(image, interpolation='nearest')

    plt.show()


if __name__ == '__main__':
    main()
