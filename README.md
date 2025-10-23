# Weighted-Recursive-Quadrant-Clustering-Model
## How the Algorithm Works

This project implements a **deterministic, geometry-driven clustering** routine that recursively partitions a 2D dataset by **quadrants** and **linear splits** anchored to a moving reference point (the “mean origin”). Conceptually, it sits between a **quadtree/KD-tree spatial partitioner** and **divisive hierarchical clustering**: splits are fast, reproducible, and easy to visualize, while still adapting to the data by re-centering on dense regions.

### High-Level Flow

1. **Load & Center**  
   - Read a DataFrame of points with columns `x` and `y` (indexed by `point`).  
   - Compute the **mean origin** using the average of all `x` and `y` values.

2. **Quadrant Split (Q4)**  
   - Partition all points into four clusters relative to the mean origin:  
     - Q1: `x > mean_x` and `y > mean_y`  
     - Q2: `x <= mean_x` and `y > mean_y`  
     - Q3: `x <= mean_x` and `y <= mean_y`  
     - Q4: `x > mean_x` and `y <= mean_y`

3. **Optional Refinement (Q8, Q16, …)**  
   - For each cluster, draw a line through the cluster centroid and the mean origin and split the cluster into two subclusters.  
   - The split line can be either the **direct line** (centroid → mean origin) or its **perpendicular**, depending on configuration.  
   - Repeating this process recursively doubles the number of clusters (for example, Q4 → Q8).

4. **Re-Centering by Cluster Weights (Adaptive Mean)**  
   - Compute a **weighted center** from the current clusters, where each cluster’s centroid is weighted by its number of points.  
   - Update the mean origin and re-split the data by quadrants using this new center.  
     This shifts the reference point toward denser regions of data.

5. **Quality & Stopping (Optional)**  
   - Each cluster can report a **mean residual** (the average distance of its points from its centroid).  
   - The process can stop when clusters are sufficiently compact, for example when the mean residual is below a threshold.

### Core Concepts

- **Clusters as DataFrames**  
  Each cluster wraps a Pandas DataFrame and maintains attributes such as its centroid, size, and helper methods for splitting.

- **Geometric Splits via Lines**  
  A split line is defined as `y = m*x + b`, where `m` and `b` are derived from the cluster centroid and the current mean origin.  
  The algorithm supports both direct and perpendicular lines for partitioning.

- **Adaptive Mean Origin**  
  After each iteration, the model can re-weight the global center using cluster sizes.  
  This acts as a global centroid update, similar to k-means but deterministic.

### Why This Approach?

- **Deterministic and Interpretable** – No random initialization or iterative reassignment.  
- **Fast Partitioning** – O(n) per level, with no global optimization loop.  
- **Easy Visualization** – Clear geometric regions that evolve logically as the model refines.

### When to Use It

- When you need a **quick, explainable spatial partitioning** of 2D data.  
- When your data benefits from **axis-aligned or linear decision boundaries**.  
- When you want a **hierarchical, recursive structure** (Q4 → Q8 → …) with adjustable compactness criteria.

### Limitations (and Mitigations)

- **Linear Boundaries Only** – Nonlinear shapes may be fragmented.  
  *Mitigation:* Increase refinement depth or hybridize with k-means on the leaf clusters.

- **Outlier Sensitivity (Mean Centering)** – Outliers can distort the global mean.  
  *Mitigation:* Use median centering or remove outliers before clustering.

- **Uniform Split Counts** – Without stopping conditions, splits double uniformly.  
  *Mitigation:* Stop splitting clusters below a residual threshold or minimum size.

### Complexity

- **Quadrant split:** O(n)  
- **Linear split per cluster:** O(n_c) per cluster  
- **One refinement level over all clusters:** O(n)  
- **Overall complexity:** O(L·n) for L refinement levels  
Memory scales with the number of active clusters.

### Practical Tips

- **Normalize** input features if `x` and `y` differ in scale.  
- **Define stopping rules** using `mean_residual()` or cluster size.  
- **Visualize** after each refinement to confirm boundary alignment.

---

**License:**  
This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.  
You are free to use, modify, and distribute this software (including commercially) provided that derivative works are also licensed under GPLv3 and source code remains publicly available.
