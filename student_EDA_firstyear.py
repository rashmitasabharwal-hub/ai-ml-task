"""student_EDA_firstyear.py


Run: python student_EDA_firstyear.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load input CSV (assumes file is in same directory)
df = pd.read_csv('student_wellbeing_dataset.csv')

# Basic cleaning (same steps as notebook): rename, normalize categories, impute medians, drop duplicates
df.rename(columns={
    'Hours_Study': 'Hours_of_Study_per_day',
    'Sleep_Hours': 'Average_Sleep_Hours',
    'Screen_Time': 'Daily_Screen_Time',
    'Attendance': 'Attendance_Percentage',
    'Extracurricular': 'Extracurricular_Activities'
}, inplace=True)

for col in ['Extracurricular_Activities','Stress_Level']:
    df[col] = df[col].astype(str).str.strip()

df['Extracurricular_Activities'] = df['Extracurricular_Activities'].str.lower().map(
    lambda x: 'Yes' if x in ['yes','y','true','1'] else ('No' if x in ['no','n','false','0'] else x.capitalize())
)

df['Stress_Level'] = df['Stress_Level'].str.lower().map(
    lambda x: 'Low' if x in ['low','l'] else ('Medium' if x in ['medium','med','m'] else ('High' if 'high' in x else x.capitalize()))
)

df = df.drop_duplicates().reset_index(drop=True)

for c in ['Hours_of_Study_per_day','Average_Sleep_Hours','Daily_Screen_Time','Attendance_Percentage','CGPA']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c].fillna(df[c].median(), inplace=True)

df['CGPA'] = df['CGPA'].clip(0,10).round(2)
df.to_csv('student_wellbeing_dataset_cleaned.csv', index=False)

# Simple regression
cat_cols = ['Extracurricular_Activities','Stress_Level']
ohe = OneHotEncoder(drop='first', sparse=False)
ohe_arr = ohe.fit_transform(df[cat_cols])
ohe_cols = ohe.get_feature_names_out(cat_cols)
ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)

X = pd.concat([df[['Hours_of_Study_per_day','Average_Sleep_Hours','Daily_Screen_Time','Attendance_Percentage']], ohe_df], axis=1)
y = df['CGPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}, R2: {r2:.4f}')

# Save a predicted vs actual plot
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0,10],[0,10], linestyle='--')
plt.xlabel('Actual CGPA'); plt.ylabel('Predicted CGPA'); plt.title('Predicted vs Actual CGPA')
plt.tight_layout(); plt.savefig('predicted_vs_actual_cgpa.png'); plt.close()
