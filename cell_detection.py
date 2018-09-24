import numpy as np
import imageio

PIXEL_THRESHOLD = 0.0

# Original image has R, G, B channels
# Shape: (X pixels, Y pixels, 3 color channels)
original_image = imageio.imread('simpleTest.png')
print(f"original shape: {original_image.shape}")

grayscale_image = np.mean(original_image, axis=2)
grayscale_image /= 255.0
print(f"grayscale shape: {grayscale_image.shape}")
print(f"max grayscale pixel: {np.max(grayscale_image)}")

def floodfill(matrix, x, y):
    # stop clause - not reinvoking for 0, only for >0
    if matrix[x][y] > 0.0:  
        matrix[x][y] = 0.0 
        #recursively invoke flood fill on all surrounding cells:
        if x > 0:
            floodfill(matrix,x-1,y)
        if x < len(matrix[y]) - 1:
            floodfill(matrix,x+1,y)
        if y > 0:
            floodfill(matrix,x,y-1)
        if y < len(matrix) - 1:
            floodfill(matrix,x,y+1)