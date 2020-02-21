import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/sgale/PycharmProjects/test/datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")
print('Data information: ', data.info())
print('Data description\n', data.describe())

last_column = data['ID']
uniqueIDs = last_column.unique()
print('Amount of unique IDs', len(uniqueIDs))


print('Size of data: ', data.shape)

# correlation = data.corr()

# print('Correlation: \n', correlation['ID'])


# Count percentage of participation on WAPs
noise = []
print('Values:\n')
for column in data:
    column = data[column]
    zeroes = 0
    ones = 0
    for value in column:
        if value == 0:
            zeroes += 1
        else:
            ones += 1
    ones_percent = ones / (zeroes + ones)
    zeroes_percent = zeroes / (zeroes + ones)
    # If the WAP is never connected add to noise list
    if ones == 0:
        noise.append(column.name)
    # print(column.name, '\t\t', 'Ones: ', ones, '\t', 'Zeroes: ', zeroes)

# Count how many WAPs are noise
print('Noise Count:', noise.__len__())
print('Noise list:', noise)


# Get a copy of the dataFrame only with useful information
usefulData = data
print('Size before removing noise:', usefulData.shape)
for uselessColumn in noise:
    usefulData.drop(uselessColumn, axis=1, inplace=True)
print('Size after removing noise: ', usefulData.shape)

usefulIdCount = []
allIdCount = []

infoId = np.zeros((len(uniqueIDs), 3))

for i in range(len(uniqueIDs)):
    result = usefulData.apply(lambda x: True if x['ID'] == uniqueIDs[i] else False, axis=1)
    count = len(result[result == True].index)
    usefulIdCount.append(count)
    infoId[i][0] = uniqueIDs[i]
    infoId[i][1] = count
    # uniqueIDs[i] = [uniqueIDs[i], count]
    break
print('Unique ID counts: \n', uniqueIDs)
plt.show()
