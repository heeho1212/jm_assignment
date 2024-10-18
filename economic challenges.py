import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 샘플 데이터 생성 - 사용자 소비 데이터 (카테고리별 지출)
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Food': [300, 450, 200, 500, 350],
    'Transport': [150, 200, 120, 300, 180],
    'Shopping': [400, 600, 300, 700, 500],
    'Entertainment': [200, 300, 150, 400, 250],
    'Others': [100, 150, 80, 200, 120]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 소비 패턴 분석을 위해 사용자별 지출 데이터를 정규화
features = ['Food', 'Transport', 'Shopping', 'Entertainment', 'Others']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# K-Means 클러스터링 모델 적용
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# 클러스터링 결과 시각화
def plot_clusters(df, features, cluster_column):
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue']
    for cluster in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster]
        plt.scatter(cluster_data[features[0]], cluster_data[features[1]],
                    color=colors[cluster], label=f'Cluster {cluster}')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('User Spending Clusters')
    plt.legend()
    plt.show()

# Food와 Shopping 항목으로 클러스터링 결과 시각화
plot_clusters(df, ['Food', 'Shopping'], 'Cluster')

# 과소비 경향 분석 및 결과 출력
def identify_overspending(df, cluster_column):
    overspending_clusters = []
    for cluster in df[cluster_column].unique():
        avg_spending = df[df[cluster_column] == cluster][features].mean().sum()
        if avg_spending > df[features].mean().sum():
            overspending_clusters.append(cluster)
    return overspending_clusters

overspending_clusters = identify_overspending(df, 'Cluster')
print(f"과소비 경향이 있는 클러스터: {overspending_clusters}")

# 과소비 사용자 식별 및 절약 추천
def recommend_saving(df, overspending_clusters, cluster_column):
    for cluster in overspending_clusters:
        overspenders = df[df[cluster_column] == cluster]
        for _, row in overspenders.iterrows():
            print(f"{row['User']}님, 과소비 경향이 있습니다. 절약을 추천드립니다.")

recommend_saving(df, overspending_clusters, 'Cluster')
