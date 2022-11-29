import os
import shutil

from matplotlib import testing



def split(folder: str, split=[0.9, 0.1, 0.2]):
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


        training_set_folder += "/fingers_"
        testing_set_folder += "/fingers_"
        validation_set_folder += "/fingers_"

        if os.path.isdir(training_set_folder + str(finger)):
            shutil.rmtree(training_set_folder + str(finger))
        if os.path.isdir(testing_set_folder + str(finger)):
            shutil.rmtree(testing_set_folder + str(finger))
        if os.path.isdir(validation_set_folder + str(finger)):
            shutil.rmtree(validation_set_folder + str(finger))
        os.mkdir(training_set_folder + str(finger))
        os.mkdir(testing_set_folder + str(finger))
        os.mkdir(validation_set_folder + str(finger))


        # I think this does not equally distribute from each finger folder.
        for file in range(0, num_training_images):
            original_filepath = folder + "/fingers_" + str(finger)
            new_filepath = training_set_folder + str(finger)
            shutil.copyfile(original_filepath, new_filepath)

        for file in range(0, num_testing_images):
            original_filepath = folder + "/fingers_" + str(finger)
            new_filepath = testing_set_folder + str(finger)
            shutil.copyfile(original_filepath, new_filepath)

        for file in range(0, num_validation_images):
            original_filepath = folder + "/fingers_" + str(finger)
            new_filepath = validation_set_folder + str(finger)
            shutil.copyfile(original_filepath, new_filepath)