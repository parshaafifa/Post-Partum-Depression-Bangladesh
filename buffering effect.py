import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load your dataset
# Ensure the CSV file is in the same folder as this script
df = pd.read_csv("C:/Users/User/Downloads/Data for Postpartum Depression Prediction in Bangladesh/Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv")

# 2. Preprocessing & Binary Mapping
# Convert the 'Result' column into a 0 or 1 for analysis
df['is_PPD'] = (df['EPDS Result'] == 'High').astype(int)

# Map the predictors into binary (0 = No Risk, 1 = Risk)
df['unplanned'] = (df['Pregnancy plan'] == 'No').astype(int)
df['joint_family'] = (df['Family type'] == 'Joint').astype(int)
df['low_support'] = (df['Recieved Support'] == 'Low').astype(int)

# Mapping 'Relationship with husband' (Bad/Poor/Neutral are grouped as 'Poor')
husband_map = {'Poor': 1, 'Bad': 1, 'Neutral': 1, 'Good': 0, 'Friendly': 0}
df['poor_husband'] = df['Relationship with husband'].map(husband_map)

# 3. Running the Interaction Models
factors = ['unplanned', 'poor_husband', 'joint_family']
final_results = []

print("--- Running Interaction Analysis ---")

for factor in factors:
    # A. Calculate the 4-Group Prevalence (%)
    # Group 0,0: No Factor, High Support
    # Group 1,0: Factor, High Support
    # Group 0,1: No Factor, Low Support
    # Group 1,1: Factor, Low Support
    groups = df.groupby([factor, 'low_support'])['is_PPD'].mean() * 100
    
    # B. Run Logistic Regression with Interaction Term (*)
    formula = f"is_PPD ~ {factor} * low_support"
    model = smf.logit(formula, data=df).fit(disp=0)
    
    # C. Extract the Interaction P-Value
    p_val = model.pvalues[f"{factor}:low_support"]
    
    # D. Store Results
    final_results.append({
        'Risk Factor': factor,
        'Low Risk (No Factor + High Supp)': round(groups.loc[0, 0], 1),
        'Variable Risk (Factor + High Supp)': round(groups.loc[1, 0], 1),
        'Support Risk (No Factor + Low Supp)': round(groups.loc[0, 1], 1),
        'High Risk (Factor + Low Supp)': round(groups.loc[1, 1], 1),
        'Interaction P-Value': round(p_val, 4)
    })

# 4. Display the Final Table
results_df = pd.DataFrame(final_results)
print("\n--- FINAL INTERACTION TABLE (Buffering Effect) ---")
print(results_df.to_string(index=False))

# 5. Save to CSV for your report
results_df.to_csv('buffering_effect_results.csv', index=False)
print("\nResults saved to 'buffering_effect_results.csv'")