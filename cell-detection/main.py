from cell_detector import CellDetector
import argparse
import time
import sys

def main():
    '''
    Runs the cell detector on a given input file and shows the time taken
    '''
    start = time.time()
    print("Running cell detector")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str,
                        help="specify input file name", required=True)
    args = parser.parse_args()

    detector = CellDetector(args.input_file)
    detector.run()

    end = time.time()
    elapsed = end - start
    print("Time elapsed: {:.3f} seconds".format(elapsed))

if __name__ == "__main__":
    main()
