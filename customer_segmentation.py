import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set style for professional-grade visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (10, 6)
})

# Harmonious, premium color palettes (custom HSL-based mappings)
PREMIUM_COLORS = ['#3A86C8', '#8338EC', '#FF006E', '#FFBE0B', '#38B000', '#FF5400']

def run_segmentation():
    # 1. Load the Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "Mall_Customers.csv")
    vis_dir = os.path.join(base_dir, "..", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run recreate_data.py first.")
        
    df = pd.read_csv(data_path)
    print("Dataset Loaded successfully. Shape:", df.shape)
    
    # 2. Exploratory Data Analysis (EDA)
    print("Generating EDA visualizations...")
    
    # 2.1 Distribution of Features
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(df['Age'], kde=True, ax=axes[0], color='#3A86C8', bins=15)
    axes[0].set_title('Age Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    
    sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1], color='#8338EC', bins=15)
    axes[1].set_title('Annual Income (k$) Distribution')
    axes[1].set_xlabel('Annual Income (k$)')
    axes[1].set_ylabel('Count')
    
    sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2], color='#FF006E', bins=15)
    axes[2].set_title('Spending Score Distribution')
    axes[2].set_xlabel('Spending Score (1-100)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'eda_distributions.png'), dpi=300)
    plt.close()
    
    # 2.2 Pairplot colored by Gender
    plt.figure(figsize=(10, 8))
    g = sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', palette={'Male': '#3A86C8', 'Female': '#FF006E'}, height=2.5)
    g.fig.suptitle('Pairwise Relationships by Gender', y=1.02)
    plt.savefig(os.path.join(vis_dir, 'eda_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. K-Means Clustering: 2D Model (Annual Income vs Spending Score)
    print("Preparing 2D K-Means (Annual Income vs Spending Score)...")
    X_2d = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Scale features
    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)
    
    # 3.1 Elbow Method & Silhouette Score to determine K
    wcss_2d = []
    silhouette_scores_2d = []
    k_range = range(2, 11)
    
    # Calculate for K=1 for WCSS elbow plot base
    wcss_1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X_2d_scaled).inertia_
    wcss_2d.append(wcss_1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_2d_scaled)
        wcss_2d.append(kmeans.inertia_)
        
        # Silhouette score requires at least 2 clusters
        score = silhouette_score(X_2d_scaled, kmeans.labels_)
        silhouette_scores_2d.append(score)
    
    # Plot Elbow Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss_2d, marker='o', linewidth=2.5, color='#8338EC')
    plt.axvline(x=5, linestyle='--', color='#FF006E', label='Optimal K = 5 (Elbow Point)')
    plt.title('Elbow Method (2D Features: Income & Spending)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'elbow_method.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores_2d, marker='s', linewidth=2.5, color='#FF5400')
    plt.axvline(x=5, linestyle='--', color='#3A86C8', label='Max Score K = 5')
    plt.title('Silhouette Score Analysis (2D Features)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_range)
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'silhouette_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 Train Optimal 2D Model (K=5)
    optimal_k_2d = 5
    kmeans_2d = KMeans(n_clusters=optimal_k_2d, random_state=42, n_init=10)
    df['Cluster_2D'] = kmeans_2d.fit_predict(X_2d_scaled)
    
    # Get cluster centroids in original scale
    centroids_2d = scaler_2d.inverse_transform(kmeans_2d.cluster_centers_)
    
    # Visualize 2D Clusters
    plt.figure(figsize=(12, 8))
    for cluster_idx in range(optimal_k_2d):
        cluster_data = df[df['Cluster_2D'] == cluster_idx]
        plt.scatter(
            cluster_data['Annual Income (k$)'], 
            cluster_data['Spending Score (1-100)'],
            s=80, 
            color=PREMIUM_COLORS[cluster_idx], 
            label=f'Segment {cluster_idx + 1}',
            edgecolors='black', 
            linewidth=0.5,
            alpha=0.85
        )
    
    plt.scatter(
        centroids_2d[:, 0], 
        centroids_2d[:, 1], 
        s=300, 
        marker='*', 
        color='gold', 
        edgecolors='black', 
        linewidth=1.5, 
        label='Centroids'
    )
    
    plt.title('Customer Segmentation (2D K-Means on Income & Spending Score)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, shadow=True)
    plt.savefig(os.path.join(vis_dir, 'clusters_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. K-Means Clustering: 3D Model (Age, Annual Income & Spending Score)
    print("Preparing 3D K-Means (Age, Annual Income, Spending Score)...")
    X_3d = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    
    scaler_3d = StandardScaler()
    X_3d_scaled = scaler_3d.fit_transform(X_3d)
    
    # Train 3D Model (Let's use K=5 or K=6 based on Silhouette Analysis, usually K=5 holds well)
    optimal_k_3d = 5
    kmeans_3d = KMeans(n_clusters=optimal_k_3d, random_state=42, n_init=10)
    df['Cluster_3D'] = kmeans_3d.fit_predict(X_3d_scaled)
    
    # 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for cluster_idx in range(optimal_k_3d):
        cluster_data = df[df['Cluster_3D'] == cluster_idx]
        ax.scatter(
            cluster_data['Age'],
            cluster_data['Annual Income (k$)'],
            cluster_data['Spending Score (1-100)'],
            s=60,
            color=PREMIUM_COLORS[cluster_idx],
            label=f'Segment {cluster_idx + 1}',
            edgecolors='black',
            linewidth=0.5,
            alpha=0.8
        )
        
    ax.set_title('Customer Segmentation in 3D Space (Age, Income, Spending Score)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.legend(frameon=True, bbox_to_anchor=(1.05, 0.8))
    
    # Adjust viewing angle for better clarity
    ax.view_init(elev=25, azim=45)
    plt.savefig(os.path.join(vis_dir, 'clusters_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Cluster Profiling & Characteristics (Focus on 2D model as it provides distinct segments)
    print("Profiling customer segments...")
    
    # Label the segments based on typical profiles:
    # Segment analysis:
    # 1. Low Income, Low Spending -> "Sensible/Sensible Spenders"
    # 2. Low Income, High Spending -> "Careless/Impulsive Spenders"
    # 3. High Income, Low Spending -> "Careful/Target Spenders"
    # 4. High Income, High Spending -> "Affluent/Premium Spenders"
    # 5. Average Income, Average Spending -> "Standard/Mid-range Spenders"
    
    # We will identify which index corresponds to which pattern programmatically:
    segment_profiles = []
    
    for i in range(optimal_k_2d):
        cluster_df = df[df['Cluster_2D'] == i]
        mean_income = cluster_df['Annual Income (k$)'].mean()
        mean_spending = cluster_df['Spending Score (1-100)'].mean()
        mean_age = cluster_df['Age'].mean()
        size = len(cluster_df)
        
        # Gender counts
        gender_counts = cluster_df['Gender'].value_counts(normalize=True).to_dict()
        female_pct = gender_counts.get('Female', 0.0) * 100
        male_pct = gender_counts.get('Male', 0.0) * 100
        
        # Define Label
        if mean_income > 70 and mean_spending > 70:
            label = "High Income, High Spending (Premium)"
        elif mean_income > 70 and mean_spending < 40:
            label = "High Income, Low Spending (Sparing/Conservative)"
        elif mean_income < 40 and mean_spending > 70:
            label = "Low Income, High Spending (Impulsive)"
        elif mean_income < 40 and mean_spending < 40:
            label = "Low Income, Low Spending (Budget-Conscious)"
        else:
            label = "Average Income, Average Spending (Balanced)"
            
        segment_profiles.append({
            "Cluster_ID": i,
            "Label": label,
            "Count": size,
            "Percentage": (size / len(df)) * 100,
            "Mean_Age": round(mean_age, 2),
            "Mean_Income_k$": round(mean_income, 2),
            "Mean_Spending_Score": round(mean_spending, 2),
            "Female_Pct": round(female_pct, 2),
            "Male_Pct": round(male_pct, 2)
        })
        
    profile_df = pd.DataFrame(segment_profiles).sort_values(by="Mean_Income_k$")
    profile_csv_path = os.path.join(vis_dir, "cluster_profiles.csv")
    profile_df.to_csv(profile_csv_path, index=False)
    print("Cluster profile saved at", profile_csv_path)
    print(profile_df)
    
    # 6. Generate Cluster Boxplots/Profile plots
    # Map label descriptions back to df
    label_map = {p['Cluster_ID']: p['Label'] for p in segment_profiles}
    df['Segment'] = df['Cluster_2D'].map(label_map)
    
    # Set a sorted order for segments based on income
    sorted_segments = profile_df['Label'].tolist()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    sns.boxplot(x='Segment', y='Age', data=df, ax=axes[0], order=sorted_segments, palette=PREMIUM_COLORS[:5])
    axes[0].set_title('Age Distribution by Segment')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=15)
    
    sns.boxplot(x='Segment', y='Annual Income (k$)', data=df, ax=axes[1], order=sorted_segments, palette=PREMIUM_COLORS[:5])
    axes[1].set_title('Annual Income Distribution by Segment')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=15)
    
    sns.boxplot(x='Segment', y='Spending Score (1-100)', data=df, ax=axes[2], order=sorted_segments, palette=PREMIUM_COLORS[:5])
    axes[2].set_title('Spending Score Distribution by Segment')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'cluster_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the clustered dataset for validation
    df.to_csv(os.path.join(base_dir, "..", "Mall_Customers_Segmented.csv"), index=False)
    print("Segmented dataset saved at Mall_Customers_Segmented.csv")

if __name__ == '__main__':
    run_segmentation()
