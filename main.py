import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tkinter import *
from tkinter import ttk

# ============================================================
# Cluster class
# ------------------------------------------------------------
# Represents a cluster of (x, y) points in a DataFrame.
# Handles geometric operations such as computing the cluster’s
# centroid, splitting the cluster into subclusters based on a
# dividing line, and creating new Cluster instances.
# ============================================================
class Cluster:
    
    def __init__(self, df, qname):
        """
        Initialize a Cluster instance.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'x' and 'y' columns.
            qname (str): Name identifier for the cluster.
        """
        self.data = df
        self.name = qname
        self.origin_x = self.data['x'].mean()  # Mean x-value (centroid)
        self.origin_y = self.data['y'].mean()  # Mean y-value (centroid)
        self.size = self.data['x'].count()     # Number of points in the cluster

    def printa(self):
        """Print a summary of the cluster’s properties."""
        string = f"{self.name}:\nx: {round(self.origin_x, 2)}\ny: {round(self.origin_y, 2)}\nsize: {self.size}\n"
        print(string)

    def get_count(self):

        return self.data.shape[0]

    def get_origin(self):
        """Return the (x, y) coordinates of the cluster’s centroid."""
        return (self.origin_x, self.origin_y)
    
    def prophase(self, center, divide_in):
        """
        Calculate the slope and intercept of a dividing line relative
        to the cluster’s origin and a given center point.

        Parameters:
            center (tuple): (x, y) coordinate used as a reference.
            divide_in (bool): If True, the line is perpendicular
                              to the line through the cluster and center.

        Returns:
            tuple: (slope, intercept)
        """
        if divide_in:
            # Create a perpendicular line
            slope = -1 / ((self.origin_y - center[1]) / (self.origin_x - center[0]))
            intercept = self.origin_y - (self.origin_x * slope)
        else:
            # Create a line through both points
            slope = (self.origin_y - center[1]) / (self.origin_x - center[0])
            intercept = self.origin_y - (self.origin_x * slope)

        return slope, intercept
    
    def metaphase(self, m, b):
        """
        Split the cluster’s data into two groups using the line y = m*x + b.

        Returns:
            tuple: (df1, df2) – DataFrames above and below the line.
        """
        df1 = self.data[m * self.data['x'] + b > self.data['y']]
        df2 = self.data[m * self.data['x'] + b <= self.data['y']]
        return df1, df2
    
    def telophase(self, df1, df2):
        """
        Create two new Cluster instances from the split DataFrames.

        Returns:
            tuple: (Cluster1, Cluster2)
        """
        c1 = Cluster(df1, f"{self.name}.1")
        c2 = Cluster(df2, f"{self.name}.2")
        return c1, c2

    def split(self, center, twice):
        """
        Perform the full cluster split:
        - Find the dividing line (prophase)
        - Divide points into two sets (metaphase)
        - Create subclusters (telophase)

        Returns:
            tuple: (Cluster1, Cluster2)
        """
        m, b = self.prophase(center, twice)
        df1, df2 = self.metaphase(m, b)
        c1, c2 = self.telophase(df1, df2)
        return c1, c2
    
    def mean_residual(self):
        """
        Calculate the mean residual distance of all points in the cluster 
        from the cluster's centroid (origin).

        This provides a measure of cluster compactness—lower values indicate 
        points are more tightly grouped around the mean.

        Returns:
            float: The average Euclidean distance of all cluster points 
                   from the cluster's (origin_x, origin_y).
        """
        mean_distance = np.sqrt((self.data['x'] - self.origin_x)**2 + (self.data['y'] - self.origin_y)**2).mean()
        return mean_distance

    

# ============================================================
# Model class
# ------------------------------------------------------------
# Handles the dataset as a whole, including importing data,
# computing distances, performing initial quadrant splits,
# further sub-dividing clusters, and plotting the results.
# ============================================================
class Model():

    def __init__(self, filepath):
        """
        Initialize a Model instance.

        Parameters:
            filepath (str): Path to the dataset CSV file.
        """
        self.full_df = None
        self.mean_origin = None
        self.temp_clusters = []

        self.import_data(filepath)

    def import_data(self, filepath):
        """Load dataset from a CSV file and compute its mean origin."""
        self.full_df = pd.read_csv(filepath, index_col="point")
        self.mean_origin = (float(self.full_df['x'].mean()), float(self.full_df['y'].mean()))

    def calc_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.

        Returns:
            float: Distance between point1 and point2.
        """
        xdiff = (point2[0] - point1[0])**2
        ydiff = (point2[1]- point1[1])**2
        distance = math.sqrt(xdiff + ydiff)
        return distance
    
    def split_quads(self):
        """
        Divide the dataset into four quadrants relative to the mean origin.
        Creates four Cluster instances (C1–C4) and stores them.
        """
        q1 = f"x > {self.mean_origin[0]} and y > {self.mean_origin[1]}"
        q2 = f"x <= {self.mean_origin[0]} and y > {self.mean_origin[1]}"
        q3 = f"x <= {self.mean_origin[0]} and y <= {self.mean_origin[1]}"
        q4 = f"x > {self.mean_origin[0]} and y <= {self.mean_origin[1]}"
        
        q1_data = self.full_df.query(q1)
        q2_data = self.full_df.query(q2)
        q3_data = self.full_df.query(q3)
        q4_data = self.full_df.query(q4)

        # Store initial quadrant clusters
        self.temp_clusters.extend([
            Cluster(q1_data, "C1"), 
            Cluster(q2_data, "C2"),
            Cluster(q3_data, "C3"),
            Cluster(q4_data, "C4")
        ])

    def weight_quads(self):
        """
        Recalculate the model's mean origin based on the weighted contribution 
        of each current cluster (quadrant).

        This method computes a new overall centroid (mean_origin) where each 
        cluster's centroid is weighted by its relative size (number of points). 
        This helps reposition the model’s central reference point to better 
        reflect the true data distribution.

        Process:
            1. Retrieve the point count for each existing cluster.
            2. Compute the proportional weight of each cluster relative to the total.
            3. Multiply each cluster's centroid by its weight and sum across all clusters.
            4. Compute the new weighted mean (mean_origin).
            5. Clear existing clusters and re-split the data using the updated origin.

        Returns:
            None
        """
        weights = []

        # Collect the number of points in each temporary cluster
        for c in self.temp_clusters:
            weights.append(c.get_count())

        # Calculate each cluster’s weight as a proportion of total points
        props = [p / sum(weights) for p in weights]
        xSum = 0
        ySum = 0

        # Compute weighted averages of x and y centroids
        for i in range(0, 4):
            coord = self.temp_clusters[i].get_origin()
            xSum += coord[0] * props[i]
            ySum += coord[1] * props[i]

        # Derive the new mean origin from weighted sums
        new_x = float(xSum / 4)
        new_y = float(ySum / 4)

        # Update the model’s central reference point
        self.mean_origin = (new_x, new_y)

        # Reset and re-split clusters based on the new mean origin
        self.temp_clusters.clear()
        self.split_quads()


        
    def split_eighth(self, divide_in=True):
        """
        Split each existing cluster into two subclusters,
        resulting in eight total clusters.

        Parameters:
            divide_in (bool): Determines whether to use perpendicular lines.
        """
        temp = []

        for c in self.temp_clusters:
            c1, c2 = c.split(self.mean_origin, divide_in)
            temp.extend([c1, c2])
        
        # Replace old clusters with the new, split ones
        self.temp_clusters.clear()
        self.temp_clusters.extend(temp)

    def plot_clusters(self):
        """Visualize all current clusters in a 2D scatter plot."""
        plt.figure()

        for cluster in self.temp_clusters:
            plt.scatter(cluster.data['x'], cluster.data['y'], label=f"{cluster.name}")

        plt.title("DataFrame split into Clusters")
        plt.xlabel("X values")
        plt.ylabel("Y values")
        plt.legend()
        plt.show()

    def print_distances(self):

        for c in self.temp_clusters:

            print(c.mean_residual())
    

# ============================================================
# Script entry point
# ------------------------------------------------------------
# Initializes the model, performs quadrant and eighth splits,
# and visualizes the final cluster arrangement.
# ============================================================
def start():
    """Main execution sequence."""
    filepath = "./data/data2.csv"
    root = Model(filepath)
    root.split_quads()
    root.weight_quads()
    root.split_eighth(True)
    root.plot_clusters()
    # root.print_distances()


if __name__ == "__main__":
    start()
