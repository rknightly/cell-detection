import numpy as np
import imageio

PIXEL_THRESHOLD = 0.0

# Original image has R, G, B channels
# Shape: (X pixels, Y pixels, 3 color channels)
original_image = imageio.imread('simpleTest.png')
print(f'original shape: {original_image.shape}')

grayscale_image = np.mean(original_image, axis=2)
grayscale_image /= 255.0
print(f'grayscale shape: {grayscale_image.shape}')
print(f'max grayscale pixel: {np.max(grayscale_image)}')

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

cells_found = []
for x in range(grayscale_image.shape[0]):
    for y in range(grayscale_image.shape[1]):
        if grayscale_image[x][y] > 0.0:
            cell_pixels, grayscale_image = check_cell(grayscale_image, x, y)
            cells_found.append(cell_pixels)
            print(f'Found cell with {len(cell_pixels)} pixels.')
