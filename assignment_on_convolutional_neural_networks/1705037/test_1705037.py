import sys
import numpy as np
import csv
from train_1705037 import *


stats_directory = "results/stats"
if not os.path.exists(stats_directory):
    os.makedirs(stats_directory)

path_to_folder = sys.argv[1]

image_filenames,input = get_images_from_directory(path_to_folder)

model = Model()
model.load()

predictions = []
#only take 32 images at a time
for i in range(0,len(input),32):
    output = model.predict(input[i:i+32,:,:,:])
    prediction = np.argmax(output, axis=1)
    predictions.append(prediction)

#append arrays in predictions list
prediction = np.concatenate(predictions)
print(prediction)
with open('results/1705037_prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename","Digit"])
    for i in range(len(prediction)):
        print(image_filenames[i],prediction[i])
        writer.writerow([image_filenames[i],prediction[i]])

