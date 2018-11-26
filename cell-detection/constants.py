# The filter threshold for the normalized grayscale image.
# Pixels at or below this value are considered noise.
FILTER_THRESHOLD = 0.3

# The threshold number for what constitutes a cell that is unusually small. 
# Cells at or below this size are considered noise.
SMALL_CELL_THRESHOLD = 15

# The width of the image that was used to define the small cell threshold.
# This is important to that images of other sizes can be processed as well.
ORIGINAL_IMAGE_WIDTH = 1125
