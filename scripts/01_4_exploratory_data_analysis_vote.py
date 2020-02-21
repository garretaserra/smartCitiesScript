import pandas as pd

# Load dataset
vote = pd.read_csv("..\datasets\house-votes-84.data", sep=",", header=None, na_values="?")
vote.columns = ['party', 'infants', 'water', 'budget', 'physician', 'salvador', 'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
vote = vote.astype('category')

# Class to the last column
cols = vote.columns.tolist()
n = int(cols.index('party'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
vote = vote[cols]


##### Attribute

print(vote.head())

# Attribute information
print(vote.info())

# Values distribution
print(vote.describe(include='all'))



##### Attribute to Attribute relationship

# Cross-tabulation
for (columnName, columnData) in vote.iteritems():
    print('Colunm Name : ', columnName)
    print(pd.crosstab(vote[columnName],vote['party']))



