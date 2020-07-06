import json
from gateNN.Implement import Implement
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("file", type=str, help="json setting file")

def main():
    args = parser.parse_args()
    with open(args.file) as fp:
        p = json.load(fp)
    im = Implement(**p)
    im.run()

if __name__ == "__main__":
    main()