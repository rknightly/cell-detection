from cell_detector import CellDetector
import sys, getopt

def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["input-file="])
    except getopt.GetoptError:
        print('detect-cells.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('detect-cells.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--input-file"):
            input_file = arg

    detector = CellDetector(input_file)
    detector.run()

if __name__ == "__main__":
    main(sys.argv[1:])