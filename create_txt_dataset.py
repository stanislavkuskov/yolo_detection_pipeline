import glob, os


dataset_path = '/home/kuskov/PycharmProjects/yolo_detection_pipeline/resources/datasets/for_training/images'

# Percentage of images to be used for the test set
percentage_test = 10

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
print(index_test)

names = glob.glob(os.path.join(dataset_path + '/*.jpg'))
print(names)
for pathAndFilename in names:

    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(title, ext)
    if counter == index_test + 1:
        counter = 1
        file_test.write(dataset_path + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(dataset_path + "/" + title + '.jpg' + "\n")
        counter = counter + 1