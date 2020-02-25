import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the data from the CSV file
data = pd.read_csv("datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")
# Print general information of the data
print('Data information: ', data.info())
print('Data description\n', data.describe())
print('Size of data: ', data.shape)

# Calculate how many unique IDs there are corresponding to unique locations
uniqueIDs = data['ID'].unique()
print('Amount of unique IDs', len(uniqueIDs))

# Count percentage of participation on WAPs on all entries
noise = []  # List of column names that are always 0
for column_name, column_data in data.iloc[:, :-1].iteritems():
    zeroes = 0
    ones = 0
    for value in column_data:
        if value == 0:
            zeroes += 1
        else:
            ones += 1
    # If the WAP is never connected add to noise list
    if ones == 0:
        noise.append(column_name)

# Print how many WAPs are noise and show which ones
print('Noise Count:', noise.__len__())
print('Noise list:', noise)

# Remove WAPs that are never connected
print('Size before removing noise:', data.shape)
for uselessColumn in noise:
    data.drop(uselessColumn, axis=1, inplace=True)
print('Size after removing noise: ', data.shape)

# Get count of unique ID
allIdCount = np.zeros(len(uniqueIDs))
for i in range(len(uniqueIDs)):
    allIdCount[i] = len((data.loc[data['ID'] == uniqueIDs[i]]).index)

# Create Data Frame with information on quantity of entries for each unique ID
locationDF = pd.DataFrame({'ID': uniqueIDs, 'Count': allIdCount}, columns=['ID', 'Count'])
print('Description of distribution of unique ID number of entries:\n', locationDF.describe())

results = pd.DataFrame()
# Calculate the correlation for each WAP
for location in uniqueIDs:
    connectivity = []
    for column_name, column_data in data.iloc[:, :-1].iteritems():
        connectivity.append((np.corrcoef((column_data == 1), (data['ID'] == location)))[0, 1])
    results[location] = connectivity

# Rename rows with according WAP names
for index, column in enumerate(data.columns[0:-1]):
    results.rename(index={index: column}, inplace=True)

# Order results data frame columns
results = results.reindex(sorted(results.columns), axis=1)

# Find the 5 best WAPs that correlate with all the locations
for index, column in enumerate(results.columns[0:-1]):
    print('\nFor ID: ', column)
    # Print information on the 5 WAPs with more correlation
    for row in results[column].nlargest(5, keep="first").iteritems():
        print('\tThe ', row[0], ' has a correlation of ', row[1])

# Get the top 5 WAPs at predicting the ID overall
print('\n\n\nThe best WAPs at predicting the locations are')
for row in results.sum(axis=1).nlargest(5, keep='first').iteritems():
    print('The ', row[0], 'with a value of ', row[1])

plt.show()
