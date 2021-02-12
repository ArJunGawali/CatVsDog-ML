import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

DataDir = os.path.join(
    '/home/arjun/Documents/Projects/ML-Projects/Cat-vs-Dog/PetImages')

Categories = ['Dog', 'Cat']

for i in Categories:
    path = os.path.join(DataDir, i)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr, cmap='gray')
        plt.show()
        break
    break

img_size = (150, 150)
new_arr = cv2.resize(img_arr, img_size)
plt.imshow(new_arr, cmap='gray')
plt.show()

training_data = []


def create_training_data():
    for i in Categories:
        path = os.path.join(DataDir, i)
        class_num = Categories.index(i)

        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr, img_size)
                training_data.append([new_arr, class_num])

            except Exception as e:
                pass


create_training_data()
print(len(training_data))


random.shuffle(training_data)

for i in training_data[:10]:
    print(i)

x = []
y = []

for feature, lable in training_data:
    x.append(feature)
    y.append(lable)

print(x[0].reshape(-1, 150, 150, 1))

pickle_out = open(
    '/home/arjun/Documents/Projects/ML-Projects/Cat-vs-Dog/x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open(
    '/home/arjun/Documents/Projects/ML-Projects/Cat-vs-Dog/.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
