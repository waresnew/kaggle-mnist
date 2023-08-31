import csv

import numpy as np

from model import get_model

input_file = open('test.csv', mode='r', newline='')
next(input_file)  # skip column labels
reader = csv.reader(input_file)
data = np.empty((28000, 28, 28, 1), dtype=np.uint8)
for i, row in enumerate(reader):
    data[i] = np.array(row, dtype=np.uint8).reshape((28, 28, 1))
input_file.close()
model = get_model()
predictions = model.predict(data)
output_file = open('output.csv', mode='w', newline='')
output_file.write('ImageId,Label\n')
for i, ans in enumerate(predictions):
    output_file.write(str(i+1) + ',' + str(np.argmax(ans))+'\n')
output_file.close()
