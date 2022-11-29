import cv2
import util
# import tensorflow as tf
import os
from split_dataset import split
import shutil


def main(directory, new_directory, args):
    if args.process:
        if not os.path.isdir(directory):
            return Exception("Not a valid directory!")

        if os.path.isdir(new_directory):
            if args.specific == None:
                shutil.rmtree(new_directory)
                os.makedirs(new_directory)
        else:
            os.makedirs(new_directory)

        log_string = ""
        
        subdirs = [f.path for f in os.scandir(directory) if f.is_dir()]
        for dir in subdirs:
            
            # create same directory in processed fingers directory
            new_dir = new_directory+"/"+dir.split("/")[-1]
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)

            subdir = dir.split("/")[-1]
            finger_number = subdir[subdir.index("_")+1:]

            if args.specific != None and int(finger_number) != int(args.specific):
                continue

            for idx, file in enumerate(os.listdir(dir)):
                img = cv2.imread(dir + "/" + file)
                try: # checks if file is of an image type
                    _ = img.shape
                except:
                    continue
                
                new_filename = "fingers_" + finger_number + "_" + str(idx) + ".jpg"

                if args.debug:
                    string = "PROCESSING FILE: {} -> {}\n".format(file, new_directory + "/fingers_" + finger_number + "/" + new_filename)
                    log_string += string
                    print(string, end="")
                
                new_img = util.hand_silhouetting(img, args)
                cv2.imwrite(new_directory + "/fingers_" + finger_number + "/" + new_filename, new_img)


    if args.split:
        split(new_directory, split=[0.9, 0.1, 0.2])

    
    if args.debug:
        print()
        with open("debug.txt", "w") as f:
            f.writelines(log_string)


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser("processing dataset for NN training")
    argument_parser.add_argument("-f", "--folder", type=str, default="fingers_unprocessed", help="Directory of unprocessed finger images")
    argument_parser.add_argument("--specific", type=int, default=None, help="Enter the finger number if you only want to process one folder")
    argument_parser.add_argument("-p", "--process", default=False, action="store_true", help="Boolean to enable processing dataset")
    argument_parser.add_argument("-d", "--debug", default=False, action="store_true", help="Boolean to enable verbosity--i.e. debugging")
    argument_parser.add_argument("-s", "--show_silhouetting", default=False, action="store_true", help="Boolean to enable showing silhouetting")
    argument_parser.add_argument("--split", default=False, action="store_true", help="Boolean to enable splitting dataset")
    args = argument_parser.parse_args()

    main(args.folder, args.folder[0:args.folder.index("_")]+"_processed", args)