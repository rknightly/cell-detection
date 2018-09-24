import numpy as np
import imageio
from matplotlib import pyplot as plt


PIXEL_THRESHOLD = 0.0


def read_image(file_name):
    # Original image has R, G, B channels
    # Shape: (X pixels, Y pixels, 3 color channels)
    original_image = imageio.imread(file_name)
    print(f'original shape: {original_image.shape}')

    grayscale_image = np.mean(original_image, axis=2)
    grayscale_image /= 255.0
    print(f'grayscale shape: {grayscale_image.shape}')
    print(f'max grayscale pixel: {np.max(grayscale_image):0.3f}')

    return grayscale_image


def filter_noise(image):
    print("Pre-processed mean: " + str(np.mean(image)))

    print(image.shape)
    mean_filled = np.sum(image) / np.count_nonzero(image)
    print("MEAN FILLED: " + str(mean_filled))

    threshold = mean_filled * 0.5
    unfilled_count = 0
    filled_count = 0

    print("THRESHOLD: " + str(threshold))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            print(image[x][y])
            if image[x][y] < threshold:
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


def detect_cells_in(matrix):
    image = read_image(file_name)
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
    # image = read_image('data/simple_examples/simpleTest.png')
    image = read_image('data/resized/testSlide1.png')
    plt.imshow(image, cmap='gray', interpolation='nearest')

    image = filter_noise(image)

    # detect_cells_in(image)
    plt.show()


if __name__ == '__main__':
    main()
