import cv2
import util
# import tensorflow as tf
import os


def main(directory, new_directory, args):
    if not os.path.isdir(directory):
        return Exception("Not a valid directory!")

    import shutil
    if os.path.isdir(new_directory):
        shutil.rmtree(new_directory)
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

    
    if args.debug:
        print()
        with open("debug.txt", "w") as f:
            f.writelines(log_string)


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser("processing dataset for NN training")
    argument_parser.add_argument("-f", "--folder", type=str, default="fingers_unprocessed", help="Directory of unprocessed finger images")
    argument_parser.add_argument("-d", "--debug", type=bool, default=False, help="Boolean to enable verbosity--i.e. debugging")
    argument_parser.add_argument("-s", "--show_silhouetting", type=bool, default=False, help="Boolean to enable showing silhouetting")
    args = argument_parser.parse_args()

    main(args.folder, args.folder[0:args.folder.index("_")]+"_processed", args)