{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8145d5f2",
   "metadata": {},
   "source": [
    "# Student Wellbeing — EDA + Linear Regression\n",
    "\n",
    "Beginner-friendly notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10ee32a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load cleaned dataset\n",
    "import pandas as pd\n",
    "clean = pd.read_csv('student_wellbeing_dataset_cleaned.csv')\n",
    "clean.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0906e4f0",
   "metadata": {},
   "source": [
    "## Linear Regression (predicting CGPA)\n",
    "We use Hours, Sleep, Screen Time, Attendance, Extracurricular, and Stress as predictors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "528a3785",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare features and train-test split\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "cat_cols = ['Extracurricular_Activities','Stress_Level']\n",
    "ohe = OneHotEncoder(drop='first', sparse=False)\n",
    "ohe_arr = ohe.fit_transform(clean[cat_cols])\n",
    "import pandas as pd\n",
    "ohe_cols = ohe.get_feature_names_out(cat_cols)\n",
    "ohe_df = pd.DataFrame(ohe_arr, columns=ohe_cols, index=clean.index)\n",
    "X = pd.concat([clean[['Hours_of_Study_per_day','Average_Sleep_Hours','Daily_Screen_Time','Attendance_Percentage']], ohe_df], axis=1)\n",
    "y = clean['CGPA']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred = lr.predict(X_test)\n",
    "\n",
    "print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))\n",
    "print('R2:', r2_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4cca431d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predicted vs Actual plot\n",
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(6,4))\n",
    "plt.scatter(y_test, y_pred, alpha=0.5)\n",
    "plt.plot([0,10],[0,10], linestyle='--')\n",
    "plt.xlabel('Actual CGPA')\n",
    "plt.ylabel('Predicted CGPA')\n",
    "plt.title('Predicted vs Actual CGPA')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c88779aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Residuals histogram\n",
    "residuals = y_test - y_pred\n",
    "plt.figure(figsize=(6,4))\n",
    "plt.hist(residuals, bins=30)\n",
    "plt.xlabel('Residual (Actual - Predicted)')\n",
    "plt.ylabel('Count')\n",
    "plt.title('Residuals Distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6200ce9c",
   "metadata": {},
   "source": [
    "## Coefficients (feature influence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6876abc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "coeffs = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)\n",
    "coeffs"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
