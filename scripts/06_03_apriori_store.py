'''
Adapted from: https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
'''

import pandas as pd  
from apyori import apriori # pip install apyori

# Load dataset
store_data = pd.read_csv(r'..\datasets\store_data.csv', header = None) 
store_data.head() 

# Data proprocessing 
# The Apriori library requires our dataset to be a list of lists
records = []  
for i in range(0, 7501): 
    records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])
    

# Support is an indication of how frequently the itemset appears in the dataset
# Support(B) = (Transactions containing (B))/(Total Transactions)
    
# Confidence is an indication of how often the rule has been found to be true
# Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)      

# Lift(A->B) refers to the increase in the ratio of sale of B to that expected if A and B were independent
# Lift(A→B) = (Confidence (A→B))/(Support (B))  
    
# Applying Apriori
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)  
association_results = list(association_rules)     

# Viewing the Results
print("NUMBER OF RULES: {}".format(len(association_results)))  
print("FIRST RULE: {}".format(association_results[0])) 


# Nice print
print("")
print("=====================================")
print("          ASSOCIATION RULES          ")
print("=====================================")
for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")