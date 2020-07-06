import json
from gateNN.Implement import Implement

def main():
    with open("setting_example.json") as fp:
        p = json.load(fp)
    im = Implement(**p)
    im.run()

if __name__ == "__main__":
    main()