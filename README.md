# MallSegments - Customer Segmentation with Clustering

A customer segmentation project using unsupervised clustering algorithms.
Goal: split customers into homogeneous groups based on income and spending behavior, so that each group can be targeted with its own marketing strategy.

## Dataset

`data.csv` contains 200 customers and 5 features:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned based on spending behavior |

## Methodology

1. **Loading and cleaning:** normalize column names, drop duplicate and missing rows, encode gender.
2. **Scaling:** standardize features with `StandardScaler` (required for distance-based algorithms).
3. **Choosing the number of clusters:** Elbow plot and Silhouette score for K from 2 to 10.
4. **Testing 10 algorithms** on two feature sets (`Income+Score` and `Age+Income+Score`):
   KMeans, Agglomerative (ward and average), GMM, Birch, Spectral, DBSCAN, OPTICS, MeanShift, AffinityPropagation.
5. **Evaluation** with three metrics:
   - Silhouette (higher is better)
   - Calinski-Harabasz (higher is better)
   - Davies-Bouldin (lower is better)
6. **Final model selection and cluster profiling.**

> Note: Algorithms such as OPTICS and DBSCAN label part of the data as "noise", and their Silhouette is computed only on non-noise points, so their scores are optimistic. For the final selection, only models that label at most 5% of the data as noise were considered.

## Results

Best model: **KMeans with 5 clusters on (Income, Score)**, Silhouette ≈ 0.555.
GMM and Agglomerative-ward reach almost the same quality. Adding Age lowered the quality (Silhouette ≈ 0.43), so it was dropped.

| Cluster | Size | Avg. age | Income (k$) | Spending score | Interpretation |
|---|---|---|---|---|---|
| 0 | 81 | 42.7 | 55.3 | 49.5 | Average customers |
| 1 | 39 | 32.7 | 86.5 | 82.1 | Target / premium customers |
| 2 | 22 | 25.3 | 25.7 | 79.4 | Young, high spenders with low income |
| 3 | 35 | 41.1 | 88.2 | 17.1 | High income, low spending (marketing opportunity) |
| 4 | 23 | 45.2 | 26.3 | 20.9 | Low income, low spending |

## Business Recommendations

- **Cluster 1:** loyalty program and exclusive offers to retain these customers.
- **Cluster 3:** targeted campaigns to convert high income into more purchases.
- **Cluster 2:** discounts and affordable products.
- **Clusters 0 and 4:** low-cost general advertising.

## Project Structure

```
clustering_project.py          Main code
data.csv                       Input data
model_comparison.csv           Comparison of all algorithms
cluster_profiles.csv           Profile of each cluster
customers_with_clusters.csv    Data with cluster labels
clusters.png                   Cluster plots (Income/Score and PCA)
k_selection_*.png              Elbow and Silhouette plots
```

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib
python clustering_project.py data.csv
```

## Future Improvements

- Try other scaling methods and feature selection.
- Tune DBSCAN/OPTICS hyperparameters more carefully.
- Validate cluster stability with bootstrap resampling.
