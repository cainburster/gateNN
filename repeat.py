import json
from gateNN.Implement import Implement
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("file", type=str, help="json setting file")
parser.add_argument("-t", '--times', type=int, help="times to excute the experiment", default=5)

def main():
    args = parser.parse_args()
    with open(args.file) as fp:
        p = json.load(fp)
    im = Implement(**p)
    im.repeat_run(args.times)

if __name__ == "__main__":
    main()