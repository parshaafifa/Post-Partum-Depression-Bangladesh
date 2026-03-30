import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
import statsmodels.api as sm
df = pd.read_csv("C:/Users/User/Downloads/Data for Postpartum Depression Prediction in Bangladesh/Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv")

print(df.shape)
print(df.head())
print(df.info())
df = df.drop(columns=['sr'])
df.isnull().sum()
import pandas as pd
import numpy as np

df['PPD'] = df['EPDS Result'].map({'High':1, 'Low':0})
total = len(df)
cases = df['PPD'].sum()

prevalence = (cases/total)*100

print("Total mothers:", total)
print("PPD cases:", cases)
print("Prevalence:", prevalence,"%")
from scipy.stats import chi2_contingency

categorical_vars = [
'Residence',
'Education Level',
'Husband\'s education level',
'Family type',
'Relationship with husband',
'Relationship with the in-laws',
'Recieved Support',
'Need for Support',
'Abuse',
'Pregnancy plan',
'Fear of pregnancy',
'Mode of delivery',
'Breastfeed',
'Newborn illness',
'Worry about newborn',
'Depression before pregnancy (PHQ2)',
'Depression during pregnancy (PHQ2)'
]

significant_vars = []

for var in categorical_vars:
    
    table = pd.crosstab(df[var], df['PPD'])
    
    chi2, p, dof, exp = chi2_contingency(table)
    
    print(var, "p-value:", p)
    
    if p < 0.05:
        significant_vars.append(var)


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("C:/Users/User/Downloads/Data for Postpartum Depression Prediction in Bangladesh/Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv")

# Preprocessing
df['outcome'] = (df['EPDS Result'] == 'High').astype(int)
df['is_joint'] = (df['Family type'] == 'Joint').astype(int)
df['unplanned'] = (df['Pregnancy plan'] == 'No').astype(int)
df['poor_husband'] = df['Relationship with husband'].isin(['Bad', 'Poor', 'Neutral']).astype(int)
df['low_support'] = df['Recieved Support'].isin(['Low', 'Medium']).astype(int)

# 1. GLM for Unadjusted Prevalence Ratio (PR)
# Poisson with Log link and Robust Standard Errors (cov_type='HC0')
formula = 'outcome ~ poor_husband' 
glm = smf.glm(formula, data=df, family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC0')
print(f"Unadjusted PR: {np.exp(glm.params['poor_husband']):.2f}")

# 2. GEE for Adjusted Prevalence Ratio (APR)
# 'groups' should be your clustering variable (e.g., Residence)
formula_adj = 'outcome ~ is_joint + unplanned + poor_husband + low_support + Age'
gee = smf.gee(formula_adj, data=df, groups=df['Residence'], 
              family=sm.families.Poisson(link=sm.families.links.Log()),
              cov_struct=sm.cov_struct.Exchangeable()).fit()

# Extract Results
results = pd.DataFrame({'APR': np.exp(gee.params), 'p-value': gee.pvalues})
print("\nGEE Adjusted Results:")
print(results.drop('Intercept'))