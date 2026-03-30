import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import chi2_contingency

# 1. LOAD DATA
# Ensure 'PPD_dataset_v2.csv' is in your script folder
df = pd.read_csv("C:/Users/User/Downloads/Data for Postpartum Depression Prediction in Bangladesh/Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv")

# 2. PREPROCESSING
# Map EPDS Result to an ordered category for the Ordinal Model
df['EPDS_Ordinal'] = pd.Categorical(df['EPDS Result'], categories=['Low', 'Medium', 'High'], ordered=True)

# Create binary target for the standard Clinical model (High Risk vs. Others)
df['is_high_risk'] = (df['EPDS Result'] == 'High').astype(int)

# Define indicators for the Risk Factors
df['is_joint'] = (df['Family type'] == 'Joint').astype(int)
df['poor_husband'] = df['Relationship with husband'].isin(['Bad', 'Poor', 'Neutral']).astype(int)
df['poor_inlaws'] = df['Relationship with the in-laws'].isin(['Bad', 'Poor', 'Neutral']).astype(int)
df['low_support'] = df['Recieved Support'].isin(['Low', 'Medium']).astype(int)
df['unplanned'] = (df['Pregnancy plan'] == 'No').astype(int)
df['abuse_reported'] = (df['Abuse'] == 'Yes').astype(int)
df['cs_delivery'] = (df['Mode of delivery'] == 'Caesarean Section').astype(int)

# ======================================================
# STEP 1: PREVALENCE BY LEVEL
# ======================================================
prevalence = df['EPDS Result'].value_counts(normalize=True).reindex(['Low', 'Medium', 'High']) * 100
print("--- PREVALENCE BY RISK LEVEL ---")
print(prevalence)
print("\n")

# ======================================================
# STEP 2: TREND ANALYSIS (Bivariate - how factors grow with severity)
# ======================================================
factors = ['is_joint', 'poor_husband', 'poor_inlaws', 'low_support', 'unplanned']
trend_table = df.groupby('EPDS Result')[factors].mean().reindex(['Low', 'Medium', 'High'])
print("--- RISK FACTOR TRENDS (Proportion in each group) ---")
print(trend_table)
print("\n")

# ======================================================
# STEP 3: MULTIPLE LOGISTIC REGRESSION (Binary: High vs. Others)
# ======================================================
# This answers: "What makes someone a high-risk clinical case?"
formula_binary = 'is_high_risk ~ is_joint + unplanned + poor_husband + poor_inlaws + low_support + abuse_reported + Age'
model_binary = logit(formula_binary, data=df).fit(disp=0)

summary_bin = model_binary.summary2().tables[1]
summary_bin['aOR'] = np.exp(summary_bin['Coef.'])
print("--- MULTIPLE LOGISTIC REGRESSION (Binary: High Risk vs Rest) ---")
print(summary_bin[['aOR', 'P>|z|']])
print("\n")

# ======================================================
# STEP 4: ORDINAL LOGISTIC REGRESSION (Spectrum: Low -> Medium -> High)
# ======================================================
# This answers: "What factors drive the overall increase in severity?"
formula_ord = 'EPDS_Ordinal ~ is_joint + unplanned + poor_husband + poor_inlaws + low_support + abuse_reported + Age'
model_ord = OrderedModel.from_formula(formula_ord, data=df, distr='logit')
res_ord = model_ord.fit(method='bfgs', disp=0)

# Extracting results (Odds Ratios)
summary_ord = res_ord.summary().tables[1]
results_ord = pd.read_html(summary_ord.as_html(), header=0, index_col=0)[0]
results_ord['aOR'] = np.exp(results_ord['coef'])

print("--- ORDINAL LOGISTIC REGRESSION (Full Spectrum Analysis) ---")
# Only showing predictors (skipping the thresholds/cutpoints)
print(results_ord.iloc[:-2][['aOR', 'P>|z|']])

# ======================================================
# STEP 5: VISUALIZATION
# ======================================================
trend_table.plot(kind='bar', figsize=(10,6))
plt.title('Distribution of Risk Factors across EPDS Severity Groups')
plt.ylabel('Prevalence (0.0 to 1.0)')
plt.legend(title='Risk Factors', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
