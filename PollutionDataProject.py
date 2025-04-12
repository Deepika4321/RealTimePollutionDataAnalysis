import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest

# Load the dataset
df = pd.read_csv(r"C:\Users\Hp\Desktop\CA2PYTHON.csv")

#Printing the first 5 to 6 rows
print(df.head())

# Summary statistics for numerical columns
print(df.describe())

# Summary statistics for categorical columns
print(df.describe(include=[object])) #tells count ,unique,top,freq

#gives concise summary of dataframe
print(df.info())

#Remove duplicates
df.drop_duplicates(inplace=True)

#missing values in each column
print("Count of missing Values in each column: ",df.isnull().sum())

#fill the missing values
df['pollutant_min']= df['pollutant_min'].fillna(df['pollutant_min'].mean())
df['pollutant_max']= df['pollutant_max'].fillna(df['pollutant_max'].mean())
df['pollutant_avg']= df['pollutant_avg'].fillna(df['pollutant_avg'].mean())

#Handled the missing data
print("Count of missing Values in complete data: ",df.isnull().sum().sum())

#OBJECTIVE 1:TOP 10 MOST POLLUTED STATES
# Compute average pollution per state
state_pollution = df.groupby("state")["pollutant_avg"].mean().reset_index()  # Reset index to keep 'state' as a column
top_states = state_pollution.nlargest(10, "pollutant_avg")  # Select the top 10 most polluted state
# Create bar plot
plt.figure(figsize=(10, 5))
sns.barplot(data=top_states,x="pollutant_avg",y="state",hue="state",palette="coolwarm",legend=False)
#plt.barh(top_states["state"], top_states["pollutant_avg"], color='g')
plt.xlabel("State", fontsize=12)
plt.ylabel("Average Pollution Level", fontsize=12)
plt.title("Top 10 Most Polluted States", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.show()

#OBJECTIVE 2: Violin Plot: Pollutant Distribution Across India

# Filter out rows with missing pollutant_avg
filtered = df.dropna(subset=["pollutant_avg"])

# Set plot style
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=filtered,
    x="pollutant_id",
    y="pollutant_avg",
    hue="pollutant_id", 
    palette="coolwarm"
)

plt.xlabel("Pollutant Type", fontsize=12)
plt.ylabel("Average Pollution Level", fontsize=12)
plt.title("Distribution of Pollution Levels by Pollutant Type Across India", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#OBJECTIVE 3
# Identify Outliers using IQR method
numerical_columns = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = ((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound))
print("Outliers detected using IQR method:\n", outliers_iqr)

# Box Plot for Outlier Detection
melted_df = df[numerical_columns].melt(var_name='Pollutant Type', value_name='Value')
# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pollutant Type', y='Value',hue ='Pollutant Type',data=melted_df, palette='Set2',legend=False)
# Add labels and title
plt.title("Box Plot for Outlier Detection")
plt.xlabel("Pollutant Type")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.show()

#OBJECTIVE 4 : Comput skewness of each numeric column
skewness=df[numerical_columns].skew()
print("Skewness of Numeric Features:\n", skewness)
#Plot histograms to visualize skewness
df[numerical_columns].hist(bins=20, color='purple' , grid=False, edgecolor="black")
plt.title("Histograms of Numeric Features in Pollution Dataset")
plt.show()


#OBJECTIVE 5: Correlation Between Different Pollutants (Heatmap)
corr_matrix = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Pollutant Values")
plt.show()

#OBJECTIVE 6: Line plot for average pollution in each state
# Group by state and calculate average pollution levels
df_statewise = df.groupby("state")["pollutant_avg"].mean().reset_index()

# Sort states by pollution level for better visualization
df_statewise = df_statewise.sort_values(by="pollutant_avg", ascending=False)
plt.figure(figsize=(16, 7))

# Line plot for pollution levels across different states
sns.lineplot(data=df_statewise,x="state", y="pollutant_avg"
    , linewidth=2.5, marker="o", markerfacecolor="black",markeredgewidth=1.5)

plt.title("Pollution Levels Across Indian States", fontsize=16, fontweight="bold", color="darkblue")
plt.xlabel("States", fontsize=14, fontweight="bold")
plt.ylabel("Average Pollution Level", fontsize=14, fontweight="bold")
plt.xticks(rotation=90)
plt.show()



#OBJECTIVE 7: Proportion of Different Pollutants (Pie Chart)
pollutant_counts = df['pollutant_id'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(pollutant_counts, labels=pollutant_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"), startangle=140)
plt.title("Proportion of Different Pollutants")
plt.show()

#OBJECTIVE 8 :Performing Z test to compare the average pollution between two states

state1 = "Delhi"
state2 = "Maharashtra"

# Extract pollutant_avg for both states

state1_pollution = df[df["state"] == state1]["pollutant_avg"].dropna()  
state2_pollution = df[df["state"] == state2]["pollutant_avg"].dropna()

# Step 2: Perform Z-Test (independent samples)
z_stat, p_value = ztest(state1_pollution, state2_pollution)

# Step 3: Set significance level and interpret the results
alpha = 0.05  # 5% significance level
if p_value < alpha:
    result = f"Reject the null hypothesis: Significant difference in pollution levels between {state1} and {state2}"
else:
    result = f"Fail to reject the null hypothesis: No significant difference in pollution levels between {state1} and {state2}"
print(result)
# Create a box plot to compare pollution levels between two states
filtered_df = df[df["state"].isin(["Delhi", "Maharashtra"])]
plt.figure(figsize=(10, 5))
sns.boxplot(x="state", y="pollutant_avg",hue="state",data=filtered_df ,palette={"Delhi": "red", "Maharashtra": "pink"},)
plt.title("Comparison of Pollution Levels (Delhi vs Maharashtra) - Z-Test")
plt.xlabel("State")
plt.ylabel("Pollutant Average")
plt.show()
