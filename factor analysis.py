import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
#pip install factor-analyzer
# 1. Load the dataset
# Ensure 'PPD_dataset_v2.csv' is in the same folder as this script
df = pd.read_csv("C:/Users/User/Downloads/Data for Postpartum Depression Prediction in Bangladesh/Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv")

# 2. Define Variables and Mappings
# These variables represent the psychological and social dimensions of your study
fa_cols = [
    'Relationship with the in-laws', 'Relationship with husband',
    'Relationship with the newborn', 'Relationship between father and newborn',
    'Feeling about motherhood', 'Recieved Support',
    'Fear of pregnancy', 'Worry about newborn',
    'Relax/sleep when newborn is tended ', # Note the trailing space in the column name
    'Relax/sleep when the newborn is asleep',
    'Angry after latest child birth',
    'Depression before pregnancy (PHQ2)', 
    'Depression during pregnancy (PHQ2)'
]

# Standardizing categorical responses into numerical scales
mappings = {
    'Relationship': {"Poor": 1, "Bad": 1, "Neutral": 2, "Good": 3, "Friendly": 3, "Very good": 4},
    'Motherhood': {"Sad": 1, "Neutral": 2, "Happy": 3},
    'Support': {"Low": 1, "Medium": 2, "High": 3},
    'Binary': {"No": 0, "Yes": 1},
    'PHQ2': {"Negative": 0, "Positive": 1}
}

# 3. Preprocessing
fa_df = df[fa_cols].copy()

# Apply mappings
fa_df['Relationship with the in-laws'] = fa_df['Relationship with the in-laws'].map(mappings['Relationship'])
fa_df['Relationship with husband'] = fa_df['Relationship with husband'].map(mappings['Relationship'])
fa_df['Relationship with the newborn'] = fa_df['Relationship with the newborn'].map(mappings['Relationship'])
fa_df['Relationship between father and newborn'] = fa_df['Relationship between father and newborn'].map(mappings['Relationship'])
fa_df['Feeling about motherhood'] = fa_df['Feeling about motherhood'].map(mappings['Motherhood'])
fa_df['Recieved Support'] = fa_df['Recieved Support'].map(mappings['Support'])
fa_df['Fear of pregnancy'] = fa_df['Fear of pregnancy'].map(mappings['Binary'])
fa_df['Worry about newborn'] = fa_df['Worry about newborn'].map(mappings['Binary'])
fa_df['Relax/sleep when newborn is tended '] = fa_df['Relax/sleep when newborn is tended '].map(mappings['Binary'])
fa_df['Relax/sleep when the newborn is asleep'] = fa_df['Relax/sleep when the newborn is asleep'].map(mappings['Binary'])
fa_df['Angry after latest child birth'] = fa_df['Angry after latest child birth'].map(mappings['Binary'])
fa_df['Depression before pregnancy (PHQ2)'] = fa_df['Depression before pregnancy (PHQ2)'].map(mappings['PHQ2'])
fa_df['Depression during pregnancy (PHQ2)'] = fa_df['Depression during pregnancy (PHQ2)'].map(mappings['PHQ2'])

# Drop rows with missing values for the analysis
fa_df_clean = fa_df.dropna()

# 4. Check for Factorability
# Bartlett's test (p < 0.05 is good) and KMO ( > 0.6 is good)
chi_square_value, p_value = calculate_bartlett_sphericity(fa_df_clean)
kmo_all, kmo_model = calculate_kmo(fa_df_clean)
print(f"Bartlett's Test p-value: {p_value}")
print(f"KMO Score: {kmo_model}")

# 5. Scree Plot to determine number of factors
fa = FactorAnalyzer(rotation=None)
fa.fit(fa_df_clean)
ev, v = fa.get_eigenvalues()

plt.scatter(range(1, fa_df_clean.shape[1]+1), ev)
plt.plot(range(1, fa_df_clean.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--')
plt.grid()
plt.show()

# 6. Run Factor Analysis
# We use 4 factors based on the eigenvalues > 1
# 'varimax' rotation makes factors easier to interpret by maximizing loading differences
n_factors = 4
fa_final = FactorAnalyzer(n_factors, rotation="varimax")
fa_final.fit(fa_df_clean)

# 7. Output Results
loadings = pd.DataFrame(fa_final.loadings_, 
                        index=fa_cols, 
                        columns=[f"Factor {i+1}" for i in range(n_factors)])

print("\n--- Factor Loadings ---")
print(loadings.round(3))

# Optional: Visualize the loadings with a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap='Blues')
plt.title('Factor Loading Heatmap')
plt.show()

# Save the factor scores (if you want to use them in GEE later)
factor_scores = fa_final.transform(fa_df_clean)
scores_df = pd.DataFrame(factor_scores, columns=[f'Factor_{i+1}_Score' for i in range(n_factors)])
scores_df.to_csv('postpartum_factor_scores.csv', index=False)
print("\nFactor scores saved to 'postpartum_factor_scores.csv'")
# 5. Scree Plot + Eigenvalues Interpretation
fa = FactorAnalyzer(rotation=None)
fa.fit(fa_df_clean)
ev, v = fa.get_eigenvalues()

# Print eigenvalues
eigen_df = pd.DataFrame({
    "Factor": range(1, len(ev)+1),
    "Eigenvalue": ev
})
print("\n--- Eigenvalues ---")
print(eigen_df)

# Count factors with eigenvalue > 1
n_factors_kaiser = sum(ev > 1)
print(f"\nNumber of factors (Eigenvalue > 1): {n_factors_kaiser}")
# Variance Explained
variance = fa_final.get_factor_variance()

variance_df = pd.DataFrame({
    "Factor": [f"Factor {i+1}" for i in range(n_factors)],
    "SS Loadings": variance[0],
    "Proportion Var": variance[1],
    "Cumulative Var": variance[2]
})

print("\n--- Variance Explained ---")
print(variance_df)
print(f"Bartlett's Test p-value: {p_value}")
print(f"KMO Score: {kmo_model}")