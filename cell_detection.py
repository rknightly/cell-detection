import numpy as np
import imageio
from matplotlib import pyplot as plt


FILTER_THRESHOLD = 0.3


def read_image(file_name):
    '''
    Reads the image from the file and converts to grayscale
    :param file_name: a str of the file name of the image
    :returns a numpy array of size (X, Y, 1) where X and Y are the image dimensions
    '''
    # Original image has R, G, B channels
    # Shape: (X pixels, Y pixels, 3 color channels)
    original_image = imageio.imread(file_name)
    print(f'original shape: {original_image.shape}')

    plt.figure(0)
    plt.imshow(original_image, interpolation='nearest')

    grayscale_image = np.mean(original_image, axis=2)
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


def check_cell(matrix, initial_x, initial_y):
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
                cell_pixels, image = check_cell(image, x, y)
                cells_found.append(cell_pixels)
                center = np.mean(cell_pixels, axis=0)
                print(f'Found cell with {len(cell_pixels)} pixels. With center: {center[0]:.1f}, {center[1]:.1f}')

    print(f'Found {len(cells_found)} cells.')


def main():
    '''
    Main function of the program.
    Reads the image from the file, displays it, filters it and displays it again.
    Finally, it detects the cells in the image and prints the locations
    '''
    # image = read_image('data/simple_examples/simpleTest.png')
    image = read_image('data/resized/testSlide1.png')
    plt.figure(1)
    plt.imshow(image, cmap='gray', interpolation='nearest')

    image = normalize(image)
    image = filter_noise(image)
    plt.figure(2)
    plt.imshow(image, cmap='gray', interpolation='nearest')


    detect_cells_in(image)
    plt.show()


if __name__ == '__main__':
    main()
