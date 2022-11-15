import os
import shutil



def main(folder: str, split=[0.9, 0.1, 0.2]):
    # split is defined to be [training, testing, validation]
    assert split[0] + split[1] == 1, "split percentages must add up to 100%!"
    assert any([x < 0 for x in split]), "split percentage must not be less than 0!"




if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser("split dataset for NN training")
    argument_parser.add_argument("-f", "--folder", type=str, default="fingers_unprocessed", help="Directory of unprocessed finger images")
    args = argument_parser.parse_args()

    main(args.folder)