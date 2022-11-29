import os
import shutil
import random



def shuffle(folder: str, batches):
    for batch in range(batches):
        for finger in range(1, 6):
            new_path = folder + '/fingers_' + str(finger) + "/"

            directory = os.listdir(folder + '/fingers_' + str(finger))
            for i in range(len(directory)):
                # we will swap files names x times, where x is the size of the listed directory
                file1 = new_path + random.choice(directory)
                file2 = new_path + random.choice(directory)
                while (file1 == file2):
                    file2 = new_path + random.choice(directory)
                file1_ext = file1[file1.rindex("."):]
                file2_ext = file2[file2.rindex("."):]
                # rename file 1 to temp
                # rename file 2 to file 1
                # rename file 1 to file 2
                file1_temp = file1[:file1.rindex("/")] + "/temp" + file1_ext
                os.rename(file1, file1_temp)
                os.rename(file2, file1[:file1.rindex(".")] + file2_ext)
                os.rename(file1_temp, file2[:file2.rindex(".")] + file1_ext)



def split(folder: str, args, split=[0.9, 0.1, 0.2]):
    # first we shuffle the entire dataset to avoid 
    shuffle(folder, args.shuffle_batches)
    if args.just_shuffle or not args.split:
        return

    # split is defined to be [training, testing, validation]
    assert split[0] + split[1] == 1, "split percentages must add up to 100%!"
    assert all([x > 0 for x in split]), "split percentage must not be less than 0!"



    training_set_folder = folder + "/training_set"
    testing_set_folder = folder + "/testing_set"
    validation_set_folder = folder + "/validation_set"

    if os.path.exists(training_set_folder):
        shutil.rmtree(training_set_folder)
    if os.path.exists(testing_set_folder):
        shutil.rmtree(testing_set_folder)
    if os.path.exists(validation_set_folder):
        shutil.rmtree(validation_set_folder)

    os.mkdir(training_set_folder)
    os.mkdir(testing_set_folder)
    os.mkdir(validation_set_folder)


    training_image_continue = 0
    testing_image_continue = 0
    validation_image_continue = 0
    


    for finger in range(1,6):
        num_images = 0
        num_training_images = 0
        num_testing_images = 0
        num_validation_images = 0
        for file in os.listdir(folder + "/fingers_" + str(finger)):
            if file.endswith(('.jpg', '.png', 'jpeg')):
                num_images += 1
        
        num_training_images = int(split[0]*num_images)
        num_testing_images = int(split[1]*num_images)
        num_validation_images = int(split[2]*num_images)


        training_set_folder += "/fingers_" + str(finger)
        testing_set_folder += "/fingers_" + str(finger)
        validation_set_folder += "/fingers_" + str(finger)

        if os.path.isdir(training_set_folder):
            shutil.rmtree(training_set_folder)
        if os.path.isdir(testing_set_folder):
            shutil.rmtree(testing_set_folder)
        if os.path.isdir(validation_set_folder):
            shutil.rmtree(validation_set_folder)
        os.mkdir(training_set_folder)
        os.mkdir(testing_set_folder)
        os.mkdir(validation_set_folder)


        for file in range(0, num_training_images):
            original_filepath = folder + "/fingers_" + str(finger) + "/fingers_" + str(finger) + "_" + str(file) + ".jpg"
            new_filepath = training_set_folder + "/fingers_" + str(finger) + "_" + str(file) + ".jpg"
            shutil.copy(original_filepath, new_filepath)

        for file in range(num_training_images, num_testing_images+num_training_images):
            original_filepath = folder + "/fingers_" + str(finger) + "/fingers_" + str(finger) + "_" + str(file) + ".jpg"
            new_filepath = testing_set_folder + "/fingers_" + str(finger) + "_" + str(file) + ".jpg"
            shutil.copy(original_filepath, new_filepath)

        for file in range(num_training_images+num_testing_images, num_validation_images+num_training_images+num_testing_images):
            original_filepath = folder + "/fingers_" + str(finger) + "/fingers_" + str(finger) + "_" + str(file) + ".jpg"
            new_filepath = validation_set_folder + "/fingers_" + str(finger) + "_" + str(file) + ".jpg"
            if not os.path.isdir(original_filepath):
                break
            shutil.copy(original_filepath, new_filepath)

        
        training_set_folder = training_set_folder[0:-10]
        testing_set_folder = testing_set_folder[0:-10]
        validation_set_folder = validation_set_folder[0:-10]