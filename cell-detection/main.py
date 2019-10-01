from cell_detector import CellDetector
import argparse
import time


def main():
    '''
    Runs the cell detector on a given input file and shows the time taken
    '''
    start = time.time()
    print("Running cell detector")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str,
                        help='specify input file name', required=True)
    parser.add_argument('-o', '--output-file', type=str, default='results.json',
                        help='specify output file name', required=False)
    parser.add_argument('--show-process', action='store_true',
                        help='Display images to show the steps of filtering'
                             ' and processing')
    args = parser.parse_args()

    detector = CellDetector(input_file=args.input_file,
                            output_file=args.output_file,
                            show_process=args.show_process)
    detector.run()

    end = time.time()
    elapsed = end - start
    print('Time elapsed: {:.3f} seconds'.format(elapsed))


if __name__ == "__main__":
    main()
