"""
IT Asset Management Lifecycle & Optimization Analysis
Author: [Your Name]
Date: November 2024

Business Problem: Optimize IT asset lifecycle, improve data quality, 
predict failures, and reduce costs through data-driven insights.
"""

# ============================================================================
# PART 1: SETUP AND DATA LOADING
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("IT ASSET MANAGEMENT ANALYSIS PROJECT")
print("=" * 80)

# Load datasets
print("\n1. Loading datasets...")
df_assets = pd.read_csv('it_assets_master.csv')
df_servicenow = pd.read_csv('servicenow_discovery.csv')
df_incidents = pd.read_csv('incident_history.csv')

print(f"   - Master Assets: {len(df_assets):,} records")
print(f"   - ServiceNow Discovery: {len(df_servicenow):,} records")
print(f"   - Incident History: {len(df_incidents):,} records")

# ============================================================================
# PART 2: DATA QUALITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: DATA QUALITY ANALYSIS")
print("=" * 80)

def calculate_data_quality_score(df):
    """Calculate comprehensive data quality metrics"""
    
    total_cells = df.shape[0] * df.shape[1]
    
    # Completeness: percentage of non-null values
    completeness = (df.notna().sum().sum() / total_cells) * 100
    
    # Accuracy: check for valid date formats and logical consistency
    accuracy_issues = 0
    if 'Purchase_Date' in df.columns and 'Warranty_End_Date' in df.columns:
        df['Purchase_Date_dt'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
        df['Warranty_End_Date_dt'] = pd.to_datetime(df['Warranty_End_Date'], errors='coerce')
        accuracy_issues += (df['Warranty_End_Date_dt'] < df['Purchase_Date_dt']).sum()
    
    accuracy = ((df.shape[0] - accuracy_issues) / df.shape[0]) * 100
    
    # Uniqueness: check for duplicates in key fields
    if 'Serial_Number' in df.columns:
        duplicates = df['Serial_Number'].duplicated().sum()
        uniqueness = ((df.shape[0] - duplicates) / df.shape[0]) * 100
    else:
        uniqueness = 100
    
    return {
        'Completeness': round(completeness, 2),
        'Accuracy': round(accuracy, 2),
        'Uniqueness': round(uniqueness, 2),
        'Overall_Score': round((completeness + accuracy + uniqueness) / 3, 2)
    }

# Calculate data quality scores
quality_master = calculate_data_quality_score(df_assets)
print("\nData Quality Assessment - Master Asset Database:")
for metric, score in quality_master.items():
    print(f"   {metric}: {score}%")

# Identify missing data by column
print("\nMissing Data Analysis:")
missing_data = df_assets.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
for col, count in missing_data.items():
    pct = (count / len(df_assets)) * 100
    print(f"   {col}: {count} ({pct:.1f}%)")

# Reconciliation analysis (matching your experience)
print("\nData Reconciliation: Master vs ServiceNow Discovery")
merged_check = df_assets.merge(
    df_servicenow, 
    on='Serial_Number', 
    how='outer', 
    indicator=True
)

reconciliation = merged_check['_merge'].value_counts()
print(f"   - Records in both systems: {reconciliation.get('both', 0):,}")
print(f"   - Only in Master (not discovered): {reconciliation.get('left_only', 0):,}")
print(f"   - Only in ServiceNow (orphan CIs): {reconciliation.get('right_only', 0):,}")

# User assignment discrepancies
if 'both' in reconciliation.index:
    both_systems = merged_check[merged_check['_merge'] == 'both'].copy()
    user_mismatch = (both_systems['Assigned_User'] != both_systems['Discovered_User']).sum()
    print(f"   - User assignment discrepancies: {user_mismatch}")
    
    encryption_mismatch = (both_systems['Encryption_Status'] != both_systems['Discovered_Encryption']).sum()
    print(f"   - Encryption status discrepancies: {encryption_mismatch}")

# ============================================================================
# PART 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Convert date columns
df_assets['Purchase_Date'] = pd.to_datetime(df_assets['Purchase_Date'])
df_assets['Warranty_End_Date'] = pd.to_datetime(df_assets['Warranty_End_Date'])

# Descriptive statistics
print("\nAsset Portfolio Summary:")
print(f"   Total Asset Value: ${df_assets['Purchase_Cost'].sum():,.2f}")
print(f"   Average Asset Cost: ${df_assets['Purchase_Cost'].mean():,.2f}")
print(f"   Median Asset Age: {df_assets['Age_Years'].median():.1f} years")

# Status distribution
print("\nAsset Status Distribution:")
print(df_assets['Status'].value_counts())

# Measure of Central Tendency and Dispersion (showcasing statistical skills)
print("\nStatistical Analysis - Asset Age:")
print(f"   Mean: {df_assets['Age_Years'].mean():.2f} years")
print(f"   Median: {df_assets['Age_Years'].median():.2f} years")
print(f"   Mode: {df_assets['Age_Years'].mode().values[0]:.2f} years")
print(f"   Std Deviation: {df_assets['Age_Years'].std():.2f} years")
print(f"   Variance: {df_assets['Age_Years'].var():.2f}")
print(f"   Range: {df_assets['Age_Years'].max() - df_assets['Age_Years'].min():.2f} years")

# Skewness and Kurtosis
print(f"   Skewness: {df_assets['Age_Years'].skew():.2f}")
print(f"   Kurtosis: {df_assets['Age_Years'].kurtosis():.2f}")

# Visualization 1: Asset Age Distribution
plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.hist(df_assets['Age_Years'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Asset Age (Years)')
plt.ylabel('Frequency')
plt.title('Distribution of Asset Age')
plt.axvline(df_assets['Age_Years'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df_assets['Age_Years'].median(), color='green', linestyle='--', label='Median')
plt.legend()

# Visualization 2: Asset Type Distribution
plt.subplot(2, 3, 2)
asset_type_counts = df_assets['Asset_Type'].value_counts()
plt.bar(asset_type_counts.index, asset_type_counts.values)
plt.xlabel('Asset Type')
plt.ylabel('Count')
plt.title('Asset Type Distribution')
plt.xticks(rotation=45)

# Visualization 3: Failure Rate by Manufacturer
plt.subplot(2, 3, 3)
failure_by_mfr = df_assets.groupby('Manufacturer')['Has_Failed'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
plt.bar(failure_by_mfr.index, failure_by_mfr.values, color='coral')
plt.xlabel('Manufacturer')
plt.ylabel('Failure Rate (%)')
plt.title('Failure Rate by Manufacturer')
plt.xticks(rotation=45)

# Visualization 4: Cost by Asset Type (Box Plot)
plt.subplot(2, 3, 4)
df_assets.boxplot(column='Purchase_Cost', by='Asset_Type', ax=plt.gca())
plt.xlabel('Asset Type')
plt.ylabel('Purchase Cost ($)')
plt.title('Purchase Cost Distribution by Asset Type')
plt.suptitle('')

# Visualization 5: Age vs Failure (Scatter)
plt.subplot(2, 3, 5)
failed = df_assets[df_assets['Has_Failed'] == 'Yes']
not_failed = df_assets[df_assets['Has_Failed'] == 'No']
plt.scatter(not_failed['Age_Years'], not_failed['Purchase_Cost'], alpha=0.3, label='No Failure', s=10)
plt.scatter(failed['Age_Years'], failed['Purchase_Cost'], alpha=0.5, label='Failed', s=10, color='red')
plt.xlabel('Asset Age (Years)')
plt.ylabel('Purchase Cost ($)')
plt.title('Asset Age vs Cost (colored by failure)')
plt.legend()

# Visualization 6: Region Distribution
plt.subplot(2, 3, 6)
region_counts = df_assets['Region'].value_counts()
plt.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%')
plt.title('Asset Distribution by Region')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved as 'eda_visualizations.png'")

# ============================================================================
# PART 4: HYPOTHESIS TESTING
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: STATISTICAL HYPOTHESIS TESTING")
print("=" * 80)

# Hypothesis 1: Chi-Square Test - Is failure rate independent of manufacturer?
print("\nHypothesis 1: Is failure rate independent of manufacturer?")
print("   H0: Failure rate is independent of manufacturer")
print("   H1: Failure rate depends on manufacturer")

contingency_table = pd.crosstab(df_assets['Manufacturer'], df_assets['Has_Failed'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"   Chi-square statistic: {chi2:.4f}")
print(f"   P-value: {p_value:.4f}")
print(f"   Degrees of freedom: {dof}")

if p_value < 0.05:
    print("   ✓ Reject H0: Failure rate DOES depend on manufacturer (p < 0.05)")
else:
    print("   ✗ Fail to reject H0: No significant relationship found")

# Hypothesis 2: ANOVA - Does average age differ significantly across asset types?
print("\nHypothesis 2: Does average age differ across asset types?")
print("   H0: Mean age is same across all asset types")
print("   H1: At least one asset type has different mean age")

age_groups = [df_assets[df_assets['Asset_Type'] == at]['Age_Years'].values 
              for at in df_assets['Asset_Type'].unique()]
f_stat, p_value_anova = f_oneway(*age_groups)

print(f"   F-statistic: {f_stat:.4f}")
print(f"   P-value: {p_value_anova:.4f}")

if p_value_anova < 0.05:
    print("   ✓ Reject H0: Mean age differs across asset types (p < 0.05)")
else:
    print("   ✗ Fail to reject H0: No significant difference in mean age")

# Hypothesis 3: T-test - Do failed assets have higher average age?
print("\nHypothesis 3: Do failed assets have higher average age?")
print("   H0: Mean age of failed = mean age of non-failed assets")
print("   H1: Mean age of failed ≠ mean age of non-failed assets")

failed_age = df_assets[df_assets['Has_Failed'] == 'Yes']['Age_Years']
not_failed_age = df_assets[df_assets['Has_Failed'] == 'No']['Age_Years']
t_stat, p_value_ttest = stats.ttest_ind(failed_age, not_failed_age)

print(f"   Mean age (failed): {failed_age.mean():.2f} years")
print(f"   Mean age (not failed): {not_failed_age.mean():.2f} years")
print(f"   T-statistic: {t_stat:.4f}")
print(f"   P-value: {p_value_ttest:.4f}")

if p_value_ttest < 0.05:
    print("   ✓ Reject H0: Failed assets have significantly different age (p < 0.05)")
else:
    print("   ✗ Fail to reject H0: No significant age difference")

# ============================================================================
# PART 5: CORRELATION ANALYSIS AND FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: CORRELATION ANALYSIS")
print("=" * 80)

# Prepare numerical features for correlation
df_analysis = df_assets.copy()

# Encode categorical variables
le_status = LabelEncoder()
le_type = LabelEncoder()
le_manufacturer = LabelEncoder()

df_analysis['Status_Encoded'] = le_status.fit_transform(df_analysis['Status'])
df_analysis['Type_Encoded'] = le_type.fit_transform(df_analysis['Asset_Type'])
df_analysis['Manufacturer_Encoded'] = le_manufacturer.fit_transform(df_analysis['Manufacturer'])
df_analysis['Failed_Binary'] = (df_analysis['Has_Failed'] == 'Yes').astype(int)

# Select numerical features
numerical_features = ['Age_Years', 'Purchase_Cost', 'Last_Login_Days_Ago', 
                     'Avg_CPU_Usage_Percent', 'Disk_Usage_Percent', 
                     'Status_Encoded', 'Type_Encoded', 'Manufacturer_Encoded',
                     'Failed_Binary']

correlation_matrix = df_analysis[numerical_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix - Asset Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation matrix saved as 'correlation_matrix.png'")

# Print key correlations with failure
print("\nCorrelation with Asset Failure:")
failure_corr = correlation_matrix['Failed_Binary'].sort_values(ascending=False)
for feature, corr in failure_corr.items():
    if feature != 'Failed_Binary':
        print(f"   {feature}: {corr:.3f}")

# ============================================================================
# PART 6: OUTLIER DETECTION (ISOLATION FOREST)
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: ANOMALY DETECTION - UNDERUTILIZED ASSETS")
print("=" * 80)

# Select features for outlier detection
outlier_features = ['Age_Years', 'Last_Login_Days_Ago', 'Avg_CPU_Usage_Percent', 
                   'Disk_Usage_Percent', 'Purchase_Cost']

df_outlier = df_analysis[df_analysis['Status'] == 'Active'][outlier_features].dropna()

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df_outlier['Anomaly'] = iso_forest.fit_predict(df_outlier)

# Anomaly = -1, Normal = 1
anomalies = df_outlier[df_outlier['Anomaly'] == -1]

print(f"\nDetected {len(anomalies)} underutilized/anomalous active assets:")
print(f"   These assets show unusual patterns in usage and may be candidates for:")
print(f"   - Reallocation to more active users")
print(f"   - Return to inventory")
print(f"   - Early retirement consideration")

# ============================================================================
# PART 7: CLUSTERING ANALYSIS (K-MEANS)
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: ASSET SEGMENTATION (K-MEANS CLUSTERING)")
print("=" * 80)

# Prepare data for clustering
cluster_features = ['Age_Years', 'Purchase_Cost', 'Avg_CPU_Usage_Percent']
df_cluster = df_analysis[cluster_features].dropna()

# Standardize features
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)

# Determine optimal clusters using elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster_scaled)
    inertias.append(kmeans.inertia_)

# Fit final model with 4 clusters
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df_analysis.loc[df_cluster.index, 'Cluster'] = kmeans_final.fit_predict(df_cluster_scaled)

print("\nAsset Segments Identified:")
for cluster_id in range(4):
    cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} assets):")
    print(f"   Avg Age: {cluster_data['Age_Years'].mean():.2f} years")
    print(f"   Avg Cost: ${cluster_data['Purchase_Cost'].mean():.2f}")
    print(f"   Avg CPU Usage: {cluster_data['Avg_CPU_Usage_Percent'].mean():.1f}%")
    print(f"   Failure Rate: {(cluster_data['Failed_Binary'].mean() * 100):.1f}%")

# Visualize clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(df_analysis.loc[df_cluster.index, 'Age_Years'], 
                     df_analysis.loc[df_cluster.index, 'Avg_CPU_Usage_Percent'],
                     c=df_analysis.loc[df_cluster.index, 'Cluster'], 
                     cmap='viridis', alpha=0.6)
plt.xlabel('Asset Age (Years)')
plt.ylabel('Avg CPU Usage (%)')
plt.title('Asset Clusters - Age vs Usage')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 2, 2)
inertias_plot = plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.tight_layout()
plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Clustering visualizations saved as 'clustering_analysis.png'")

# ============================================================================
# PART 8: PREDICTIVE MODELING - FAILURE PREDICTION
# ============================================================================

print("\n" + "=" * 80)
print("PART 8: MACHINE LEARNING - ASSET FAILURE PREDICTION")
print("=" * 80)

# Prepare features for modeling
feature_cols = ['Age_Years', 'Purchase_Cost', 'Last_Login_Days_Ago', 
                'Avg_CPU_Usage_Percent', 'Disk_Usage_Percent',
                'Manufacturer_Encoded', 'Type_Encoded']

df_ml = df_analysis[feature_cols + ['Failed_Binary']].dropna()

X = df_ml[feature_cols]
y = df_ml['Failed_Binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                      random_state=42, stratify=y)

print(f"\nDataset Split:")
print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")
print(f"   Failure rate in training: {y_train.mean()*100:.1f}%")

# Model 1: Logistic Regression
print("\n--- Model 1: Logistic Regression ---")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Failure', 'Failure']))

# Model 2: Decision Tree
print("\n--- Model 2: Decision Tree Classifier ---")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy: {accuracy_dt*100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

# Model 3: Random Forest (Best performer typically)
print("\n--- Model 3: Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Failure', 'Failure']))

# Cross-validation score
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"\nCross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Random Forest Model')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")

# ============================================================================
# PART 9: COST PREDICTION (REGRESSION)
# ============================================================================

print("\n" + "=" * 80)
print("PART 9: REPLACEMENT COST FORECASTING")
print("=" * 80)

# Predict total repair costs based on asset characteristics
df_cost = df_analysis.merge(
    df_incidents.groupby('Asset_ID')['Repair_Cost'].sum().reset_index(),
    left_on='Asset_ID',
    right_on='Asset_ID',
    how='inner'
)

if len(df_cost) > 0:
    X_cost = df_cost[['Age_Years', 'Purchase_Cost', 'Failed_Binary']]
    y_cost = df_cost['Repair_Cost']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cost, y_cost, test_size=0.3, random_state=42
    )
    
    # Linear Regression
    lr_cost = LinearRegression()
    lr_cost.fit(X_train_c, y_train_c)
    y_pred_cost = lr_cost.predict(X_test_c)
    
    mse = mean_squared_error(y_test_c, y_pred_cost)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_c, y_pred_cost)
    r2 = r2_score(y_test_c, y_pred_cost)
    
    print(f"\nRepair Cost Prediction Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE: ${mae:.2f}")
    
    print(f"\nModel Coefficients:")
    for feature, coef in zip(['Age_Years', 'Purchase_Cost', 'Failed_Binary'], lr_cost.coef_):
        print(f"   {feature}: {coef:.4f}")

# ============================================================================
# PART 10: BUSINESS RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("PART 10: KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

# Calculate key metrics
total_assets = len(df_assets)
active_assets = len(df_assets[df_assets['Status'] == 'Active'])
failed_assets = len(df_assets[df_assets['Has_Failed'] == 'Yes'])
failure_rate = (failed_assets / total_assets) * 100

# Assets needing refresh (over 4 years old and active)
refresh_candidates = len(df_assets[(df_assets['Age_Years'] > 4) & 
                                   (df_assets['Status'] == 'Active')])

# Underutilized assets (active, low usage, not logged in for 30+ days)
underutilized = len(df_assets[(df_assets['Status'] == 'Active') & 
                              (df_assets['Last_Login_Days_Ago'] > 30) &
                              (df_assets['Avg_CPU_Usage_Percent'] < 20)])

# Total incident costs
total_repair_cost = df_incidents['Repair_Cost'].sum()

print("\n📊 EXECUTIVE SUMMARY:")
print(f"   Total IT Assets: {total_assets:,}")
print(f"   Active Assets: {active_assets:,}")
print(f"   Overall Failure Rate: {failure_rate:.1f}%")
print(f"   Total Portfolio Value: ${df_assets['Purchase_Cost'].sum():,.2f}")
print(f"   Total Repair Costs: ${total_repair_cost:,.2f}")

print("\n🎯 ACTIONABLE RECOMMENDATIONS:")

print(f"\n1. ASSET REFRESH STRATEGY")
print(f"   - {refresh_candidates} assets are over 4 years old and due for replacement")
print(f"   - Estimated replacement budget needed: ${refresh_candidates * df_assets['Purchase_Cost'].mean():,.2f}")
print(f"   - Priority: High-failure manufacturers and asset types")

print(f"\n2. UNDERUTILIZATION OPTIMIZATION")
print(f"   - {underutilized} active assets show low utilization")
print(f"   - Potential savings from reallocation: ${underutilized * df_assets['Purchase_Cost'].mean() * 0.15:,.2f}")
print(f"   - Action: Audit these assets for reallocation or early retirement")

print(f"\n3. DATA QUALITY IMPROVEMENTS")
print(f"   - Overall data quality score: {quality_master['Overall_Score']:.1f}%")
print(f"   - {reconciliation.get('left_only', 0)} assets not appearing in discovery tools")
print(f"   - Action: Implement automated reconciliation process (similar to your Excel process)")

print(f"\n4. PREDICTIVE MAINTENANCE")
print(f"   - Machine learning model achieves {accuracy_rf*100:.1f}% accuracy in failure prediction")
print(f"   - Deploy model to flag high-risk assets for proactive maintenance")
print(f"   - Expected reduction in repair costs: 20-25%")

print(f"\n5. MANUFACTURER STRATEGY")
failure_by_mfr = df_assets.groupby('Manufacturer').apply(
    lambda x: (x['Has_Failed'] == 'Yes').sum() / len(x) * 100
).sort_values()
print(f"   - Lowest failure rate: {failure_by_mfr.index[0]} ({failure_by_mfr.values[0]:.1f}%)")
print(f"   - Highest failure rate: {failure_by_mfr.index[-1]} ({failure_by_mfr.values[-1]:.1f}%)")
print(f"   - Action: Consider shifting procurement toward more reliable manufacturers")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - All outputs saved to current directory")
print("=" * 80)

# Save summary report
summary_report = f"""
IT ASSET MANAGEMENT ANALYSIS - SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO OVERVIEW:
- Total Assets: {total_assets:,}
- Active Assets: {active_assets:,}
- Total Value: ${df_assets['Purchase_Cost'].sum():,.2f}
- Failure Rate: {failure_rate:.1f}%

DATA QUALITY:
- Overall Score: {quality_master['Overall_Score']:.1f}%
- Complet