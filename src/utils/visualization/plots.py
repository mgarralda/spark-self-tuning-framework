# -----------------------------------------------------------------------------
#  Project: Spark Self-Tuning Framework (STL-PARN-ILS-TS-BO)
#  File: yoro_perf_model_runner.py
#  Copyright (c) 2025 Mariano Garralda Barrio
#  Affiliation: Universidade da Coruña
#  SPDX-License-Identifier: CC-BY-NC-4.0 OR LicenseRef-Commercial
#
#  Associated publication:
#    "A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous Spark tuning"
#    Mariano Garralda Barrio, Verónica Bolón Canedo, Carlos Eiras Franco
#    Universidade da Coruña, 2025.
#
#  Academic & research use: CC BY-NC 4.0
#    https://creativecommons.org/licenses/by-nc/4.0/
#  Commercial use: requires prior written consent.
#    Contact: mariano.garralda@udc.es
#
#  Distributed on an "AS IS" basis, without warranties or conditions of any kind.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import combinations
from scipy.stats import binned_statistic_2d
from pandas.plotting import parallel_coordinates


def plot_config_density(self, x_new_records, ucb_values):
    """
    Visualizes the density of configurations and their corresponding UCB values.

    x_new_records: list of candidate configurations.
    ucb_values: list of UCB values corresponding to each configuration.
    """

    # Convert to DataFrame
    df = pd.DataFrame(x_new_records,
                      columns=["cores", "memory", "dim3", "dim4", "dim5", "dim6"])
    df["LCB"] = ucb_values

    # Pairwise KDE plots for each configuration dimension
    plt.figure(figsize=(10, 6))
    for i, column in enumerate(df.columns[:-1]):  # Exclude LCB column
        sns.kdeplot(df[column], fill=True, label=column, alpha=0.5)

    plt.title("Density of Configurations")
    plt.xlabel("Configuration Values")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Highlight density per dimension with LCB
    for column in df.columns[:-1]:  # Exclude LCB column
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=df[column], hue=df["LCB"], fill=True, palette="viridis", alpha=0.5)
        plt.title(f"Density of {column} Configurations by LCB")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.show()

def plot_config_heatmap(self, x_new_records, ucb_values):
    """
    Heatmap to visualize frequently visited configurations across dimensions,
    and compute the mean LCB values for all dimensions.
    """

    # Map bounds to meaningful dimension names
    dimension_names = [
        "driver_cores",
        "driver_memory_gb",
        "executor_cores",
        "executor_instances",
        "executor_memory_gb",
        "sql_shuffle_partitions"
    ]

    # Convert to DataFrame
    df = pd.DataFrame(x_new_records, columns=dimension_names)
    df["LCB"] = ucb_values

    # Compute mean LCB for each dimension
    mean_lcb_per_dimension = df.groupby(dimension_names).mean(numeric_only=True)["LCB"]
    print("\nMean LCB per configuration dimension:")
    for dim in dimension_names:
        mean_lcb = df.groupby(dim)["LCB"].mean()
        print(f"- {dim}: {mean_lcb.mean():.2f} (overall mean)")

    # Choose two key dimensions for a heatmap (e.g., "executor_cores" vs "executor_memory_gb")
    heatmap_data = df.pivot_table(index="executor_cores",
                                  columns="executor_memory_gb",
                                  values="LCB",
                                  aggfunc="mean")

    plt.figure(figsize=(8, 6))
    # Use a colormap where darker colors indicate lower values (e.g., "mako_r")
    sns.heatmap(heatmap_data, annot=True, cmap="mako_r", cbar_kws={"label": "Mean LCB"}, fmt=".2f")
    plt.title("Heatmap of Executor Cores vs Executor Memory GB (Lower LCB is Better)")
    plt.xlabel("Executor Memory (GB)")
    plt.ylabel("Executor Cores")
    plt.show()

    return mean_lcb_per_dimension

def analyze_candidate_configs(self, x_new_records, ucb_values):
    """
    x_new_records: list of numpy arrays representing candidate configurations (6D).
    ucb_values: list of LCB values corresponding to x_new_records.
    """

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(x_new_records,
                      columns=["cores", "memory", "dim3", "dim4", "dim5", "dim6"])
    df["LCB"] = ucb_values

    # Remove rows where LCB is NaN or Inf
    df = df[~df["LCB"].isna()]  # drop NaN
    df = df[~np.isinf(df["LCB"])]  # drop Inf

    # Sort by LCB so color bins make sense
    df_sorted = df.sort_values("LCB").reset_index(drop=True)

    # Bin LCB values into discrete categories
    df_sorted["LCB_Bin"] = pd.cut(df_sorted["LCB"], bins=5, labels=False)

    # --- Parallel Coordinates ---
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df_sorted, class_column="LCB_Bin", colormap=plt.cm.viridis)
    plt.title("Parallel Coordinates - Configs Colored by LCB Bins")
    plt.xlabel("Dimensions")
    plt.ylabel("Values")
    plt.show()

    # --- Pairwise Scatter Plot ---
    sns.pairplot(
        df,
        vars=["cores", "memory", "dim3", "dim4", "dim5", "dim6"],
        diag_kind="hist",
        corner=True,
        plot_kws={"c": df["LCB"], "cmap": "viridis"}
    )
    plt.suptitle("Pairwise Plots - Configurations Colored by LCB", y=1.02)
    plt.show()

def visualize_ucb_behavior_(self, candidate_space, mu_values, sigma_values, alpha_values, ucb_values):
    """
    Visualize the behavior of UCB and its components over the candidate space.

    :param candidate_space: Array of candidate configurations.
    :param mu_values: Predicted mean values for the candidates.
    :param sigma_values: Scaled uncertainty values for the candidates.
    :param alpha_values: Dynamic alpha values for the candidates.
    :param ucb_values: Computed UCB values for the candidates.
    """

    candidates = range(len(candidate_space))

    plt.figure(figsize=(12, 8))

    # Plot mu
    plt.plot(candidates, mu_values, label=r"$\mu$ (Mean Prediction)", color='blue', linestyle='-', linewidth=2)

    # Plot scaled sigma
    plt.plot(candidates, sigma_values, label=r"$\sigma$ (Uncertainty)", color='orange', linestyle='--', linewidth=2)

    # Plot alpha
    plt.plot(candidates, alpha_values, label=r"$\alpha$ (Dynamic Weight)", color='green', linestyle='-.', linewidth=2)

    # Plot UCB values
    plt.plot(candidates, ucb_values, label="UCB Value", color='red', linestyle='-', linewidth=3)

    # Highlight max/min UCB
    max_ucb_index = np.argmax(ucb_values)
    plt.axvline(max_ucb_index, color='purple', linestyle=':', linewidth=2, label=f"Max UCB at {max_ucb_index}")

    plt.title("UCB Behavior Across Candidate Configurations", fontsize=16)
    plt.xlabel("Candidate Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def visualize_ucb_behavior(self, candidate_space, mu_values, sigma_values, alpha_values, ucb_values):
    """
    Visualize the behavior of UCB and its components over the candidate space using Seaborn.

    :param candidate_space: Array of candidate configurations.
    :param mu_values: Predicted mean values for the candidates.
    :param sigma_values: Scaled uncertainty values for the candidates.
    :param alpha_values: Dynamic alpha values for the candidates.
    :param ucb_values: Computed UCB values for the candidates.
    """

    sns.set(style="whitegrid", context="talk")

    candidates = range(len(candidate_space))
    min_ucb_index = np.argmin(ucb_values)
    min_mu_index = np.argmin(mu_values)
    max_sigma_index = np.argmax(sigma_values)

    plt.figure(figsize=(14, 8))

    # Plot mu
    sns.lineplot(x=candidates, y=mu_values, label=r"$\mu$ (Mean Prediction)",
                 color="blue", linewidth=2, linestyle="-")

    # Plot scaled sigma
    sns.lineplot(x=candidates, y=sigma_values, label=r"$\sigma$ (Uncertainty)",
                 color="green", linewidth=2, linestyle="-")

    # Plot alpha
    # sns.lineplot(x=candidates, y=alpha_values, label=r"$\alpha$ (Dynamic Weight)",
    #              color="green", linewidth=2, linestyle="-.")

    # Plot UCB values
    sns.lineplot(x=candidates, y=ucb_values, label="LCB Value",
                 color="red", linewidth=3, linestyle="-")

    # Highlight max Mu
    plt.axvline(
        min_mu_index,
        color="blue",
        linestyle=":",
        linewidth=4,
        # label=f"Best mu(exploitation) at Index {min_ucb_index} (Value: {mu_values[min_mu_index]:.2f})"
    )

    # Highlight max Sigma
    plt.axvline(
        max_sigma_index,
        color="green",
        linestyle=":",
        linewidth=4,
        # label=f"Best sigma(exploration) at Index {max_sigma_index} (Value: {sigma_values[max_sigma_index]:.2f})"
    )

    # Highlight max Uncertainty
    plt.axvline(
        min_ucb_index,
        color="red",
        linestyle=":",
        linewidth=4,
        # label=f"Best LCB at Index {min_ucb_index} (Value: {ucb_values[min_ucb_index]:.2f})"
    )

    # # Add annotations for max Uncertainty
    # plt.annotate(
    #     f"Max Uncertainty: {ucb_values[min_ucb_index]:.2f}",
    #     xy=(min_ucb_index, ucb_values[min_ucb_index]),
    #     xytext=(
    #         min_ucb_index + 10,  # Adjust horizontal offset for clarity
    #         ucb_values[min_ucb_index] + 0.1 * np.max(ucb_values)  # Vertical offset
    #     ),
    #     arrowprops=dict(facecolor="purple", arrowstyle="->", lw=1.5),  # Arrow style and thickness
    #     fontsize=12,
    #     color="purple"
    # )

    plt.title("LCB Behavior Across Candidate Configurations", fontsize=18, weight="bold")
    plt.xlabel("Candidate Index", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend(fontsize=12, loc="lower left", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

def visualize_acquisition_values(self, candidate_space, acq_values):
    """
    Visualize acquisition values using histogram, scatter plot, and pairwise heatmap.

    :param candidate_space: Array of candidate configurations (NxD, where N is the number of samples and D is the number of dimensions).
    :param acq_values: Array of acquisition function values (length N).
    """

    def plot_density_curve_with_max():
        """
        Plot the density curve of acquisition values and mark the maximum value with a vertical line.
        Debugging added to check data ranges.
        """
        clean_acq_values = np.array(acq_values)[~np.isnan(acq_values)]
        max_value = np.nanmax(clean_acq_values)  # Get the maximum value

        # print(f"Cleaned acquisition values range: {np.min(clean_acq_values)} to {np.max(clean_acq_values)}")
        # print(f"Maximum acquisition value: {max_value}")

        plt.figure(figsize=(10, 6))
        sns.kdeplot(clean_acq_values, shade=True, color='skyblue', linewidth=2, bw_adjust=0.6)
        # plt.axvline(max_value, color='red', linestyle='--', linewidth=2, label=f"Max Value: {max_value:.2f}")
        plt.title("Density Curve of Acquisition Values")
        plt.xlabel("Adaptive Lower Confident Bound Values")
        plt.ylabel("Density")
        # plt.legend()
        plt.grid(True)
        plt.show()

    def plot_density_curve():
        """
        Plot only the density curve of acquisition values using Seaborn.

        :param acq_values: Array of acquisition values.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(acq_values, shade=True, color='skyblue', linewidth=2)
        plt.title("Density Curve of Acquisition Values")
        plt.xlabel("Acquisition Values")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    # Plot 1: Histogram of Acquisition Values
    def plot_histogram():
        plt.figure(figsize=(10, 6))
        plt.hist(acq_values, bins=20, color='skyblue', edgecolor='black')
        plt.title("Histogram of Acquisition Values")
        plt.xlabel("Acquisition Values")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    # Plot 2: Scatter Plot of Acquisition Values vs Dimension 0
    def plot_scatter():
        plt.figure(figsize=(10, 6))
        plt.scatter(candidate_space[:, 0], acq_values, c='blue', alpha=0.6, label='Acquisition Values')
        plt.title("Scatter Plot: Acquisition Values vs First Dimension")
        plt.xlabel("Dimension 0")
        plt.ylabel("Acquisition Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot 3: Heatmap of Pairwise Dimensions (Mean Aggregation)
    def plot_heatmap():
        num_dimensions = candidate_space.shape[1]
        heatmap = np.zeros((num_dimensions, num_dimensions))

        for i in range(num_dimensions):
            for j in range(num_dimensions):
                if i < j:
                    statistic, _, _, _ = binned_statistic_2d(
                        candidate_space[:, i], candidate_space[:, j], acq_values, statistic="mean", bins=10
                    )
                    value = np.nanmean(statistic)
                    heatmap[i, j] = value
                    heatmap[j, i] = value

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Mean Acquisition Value'})
        plt.title("Heatmap of Pairwise Dimensions")
        plt.xlabel("Dimension")
        plt.ylabel("Dimension")
        plt.xticks(ticks=np.arange(num_dimensions) + 0.5, labels=[f"Dim {i}" for i in range(num_dimensions)], rotation=45)
        plt.yticks(ticks=np.arange(num_dimensions) + 0.5, labels=[f"Dim {i}" for i in range(num_dimensions)], rotation=0)
        plt.show()

    # Call the plotting functions
    # plot_histogram()
    # plot_density_curve()
    plot_density_curve_with_max()
    # plot_scatter()
    # plot_heatmap()

def plot_pairwise_heatmap_summary(self, candidate_space, cq_values, acquisition="UCB", bins=5, aggregation="std"):
    """
    Create a heatmap summarizing pairwise interactions for all dimensions.

    :param candidate_space: Array of candidate configurations.
    :param cq_values: Array of acquisition function values (may contain NaN).
    :param acquisition: Name of the acquisition function (for title).
    :param bins: Number of bins for the heatmap grid.
    :param aggregation: Aggregation method ("mean", "sum", "max").
    """

    num_dimensions = candidate_space.shape[1]
    heatmap = np.zeros((num_dimensions, num_dimensions))

    # Loop over all dimension pairs
    for dim1, dim2 in combinations(range(num_dimensions), 2):
        x = candidate_space[:, dim1]
        y = candidate_space[:, dim2]

        # Bin data for the pair (dim1, dim2)
        statistic, _, _, _ = binned_statistic_2d(
            x, y, cq_values, statistic=aggregation, bins=bins
        )

        # Handle NaN values in the bins
        if aggregation == "mean":
            # value = np.nanmean(statistic)  # Mean ignoring NaN
            value = np.nanmedian(statistic)
        elif aggregation == "sum":
            value = np.nansum(statistic)  # Sum ignoring NaN
        elif aggregation == "max":
            value = np.nanmax(statistic)  # Max ignoring NaN
        elif aggregation == "std":
            value = np.nanstd(statistic)  # Standard deviation ignoring NaN
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        # Debugging: Print values for verification
        print(f"Aggregation ({aggregation}) for Dim {dim1} vs Dim {dim2}: {value}")

        # Assign aggregated value to the heatmap
        heatmap[dim1, dim2] = value
        heatmap[dim2, dim1] = value  # Symmetric matrix

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="viridis", origin="lower", interpolation="nearest")
    plt.colorbar(label=f"{aggregation.capitalize()} Acquisition Value")
    plt.title(f"Pairwise Heatmap Summary ({acquisition})")
    plt.xlabel("Dimension")
    plt.ylabel("Dimension")
    plt.xticks(range(num_dimensions), [f"Dim {i}" for i in range(num_dimensions)])
    plt.yticks(range(num_dimensions), [f"Dim {i}" for i in range(num_dimensions)])
    plt.grid(False)
    plt.show()

def plot_pairwise_heatmap_summary_(self, candidate_space, acq_values, acquisition="UCB", aggregation="mean"):
    """
    Create a heatmap summarizing pairwise interactions for all dimensions.

    :param candidate_space: Array of candidate configurations.
    :param acq_values: Array of acquisition function values.
    :param acquisition: Name of the acquisition function (for title).
    :param aggregation: Aggregation method ("mean", "sum", "max").
    """

    num_dimensions = candidate_space.shape[1]
    heatmap = np.zeros((num_dimensions, num_dimensions))

    # Aggregate acquisition values for each pair of dimensions
    for dim1, dim2 in combinations(range(num_dimensions), 2):
        x = candidate_space[:, dim1]
        y = candidate_space[:, dim2]

        if aggregation == "mean":
            value = np.mean(acq_values)
        elif aggregation == "sum":
            value = np.sum(acq_values)
        elif aggregation == "max":
            value = np.max(acq_values)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        # Symmetric assignment in heatmap
        heatmap[dim1, dim2] = value
        heatmap[dim2, dim1] = value

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="viridis", origin="lower", interpolation="nearest")
    plt.colorbar(label=f"{aggregation.capitalize()} Acquisition Value")
    plt.title(f"Pairwise Heatmap Summary ({acquisition})")
    plt.xlabel("Dimension")
    plt.ylabel("Dimension")
    plt.xticks(range(num_dimensions), [f"Dim {i}" for i in range(num_dimensions)])
    plt.yticks(range(num_dimensions), [f"Dim {i}" for i in range(num_dimensions)])
    plt.grid(False)
    plt.show()

def plot_pairplot(self, candidate_space, acq_values, acquisition="UCB"):

    """
    Create a pairplot representation of all pairwise dimension interactions.

    :param candidate_space: Array of candidate configurations.
    :param acq_values: Array of acquisition function values.
    :param acquisition: Name of the acquisition function (for plot title).
    """
    # Convert candidate space and acquisition values into a DataFrame
    num_dimensions = candidate_space.shape[1]
    data = pd.DataFrame(candidate_space, columns=[f"Dim {i}" for i in range(num_dimensions)])
    data["Acquisition Value"] = acq_values

    # Create the pair plot
    pair_plot = sns.pairplot(
        data,
        diag_kind="kde",  # Kernel density estimation on the diagonal
        palette="viridis",
        hue="Acquisition Value",
        corner=True  # Only lower triangle for compactness
    )
    pair_plot.fig.suptitle(f"Pairwise Scatter Plot Matrix ({acquisition})", y=1.02, fontsize=16)
    plt.show()

def plot_all_pairwise_heatmaps(self, candidate_space, acq_values, acquisition="UCB", bins=20, aggregation="mean"):
    """
    Create a single plot with heatmaps for all pairs of dimensions.

    :param candidate_space: Array of candidate configurations.
    :param acq_values: Array of acquisition function values.
    :param acquisition: Name of the acquisition function (for plot title).
    :param bins: Number of bins for the heatmap grid.
    :param aggregation: Aggregation method ("mean", "sum", "max").
    """


    num_dimensions = candidate_space.shape[1]
    pairs = list(combinations(range(num_dimensions), 2))  # Generate all pairs of dimensions

    # Determine grid size for subplots
    num_pairs = len(pairs)
    grid_size = int(np.ceil(np.sqrt(num_pairs)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
    axes = axes.flatten()

    for idx, (dim1, dim2) in enumerate(pairs):
        x = candidate_space[:, dim1]
        y = candidate_space[:, dim2]

        # Binned statistic for the heatmap
        statistic, x_edges, y_edges, _ = binned_statistic_2d(
            x, y, acq_values, statistic=aggregation, bins=bins
        )

        ax = axes[idx]
        heatmap = ax.imshow(
            statistic.T, origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            aspect="auto", cmap="viridis"
        )
        ax.set_title(f"Dims {dim1} vs {dim2}")
        ax.set_xlabel(f"Dim {dim1}")
        ax.set_ylabel(f"Dim {dim2}")
        fig.colorbar(heatmap, ax=ax, orientation="vertical")

    # Hide empty subplots if any
    for idx in range(len(pairs), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Pairwise Heatmaps of Acquisition Function ({acquisition})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_acquisition_heatmap(self, candidate_space, acq_values, acquisition="UCB"):

    num_dimensions = candidate_space.shape[1]

    # Iterate through all pairs of dimensions
    for dim1 in range(num_dimensions):
        for dim2 in range(dim1 + 1, num_dimensions):
            # Fix remaining dimensions to their mean
            fixed_values = candidate_space.mean(axis=0)
            fixed_values[dim1] = None  # Allow these to vary
            fixed_values[dim2] = None

            # Generate a grid for dim1 and dim2
            grid_x, grid_y = np.meshgrid(
                np.linspace(candidate_space[:, dim1].min(), candidate_space[:, dim1].max(), 100),
                np.linspace(candidate_space[:, dim2].min(), candidate_space[:, dim2].max(), 100)
            )

            # Interpolate acquisition values
            points = candidate_space[:, [dim1, dim2]]
            grid_z = griddata(points, acq_values, (grid_x, grid_y), method="linear")

            # Plot the heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(grid_z, extent=(
                candidate_space[:, dim1].min(), candidate_space[:, dim1].max(),
                candidate_space[:, dim2].min(), candidate_space[:, dim2].max()
            ), origin="lower", aspect="auto", cmap="viridis")
            plt.colorbar(label="Acquisition Function Value")
            plt.xlabel(f"Dimension {dim1}")
            plt.ylabel(f"Dimension {dim2}")
            plt.title(f"Acquisition Function Heatmap ({acquisition})\nDims {dim1} vs {dim2}")
            plt.grid(False)
            plt.show()

def plot_acquisition_function(self, candidate_space, acq_values, acquisition="UCB"):



    # Plot for 1D search space
    if candidate_space.shape[1] == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(candidate_space, acq_values, label=f"{acquisition} values")
        plt.xlabel("Search Space")
        plt.ylabel("Acquisition Function Value")
        plt.title(f"Acquisition Function ({acquisition}) Over Search Space")
        plt.legend()
        plt.grid()
        plt.show()

    # Plot for 2D search space
    elif candidate_space.shape[1] == 2:

        x = candidate_space[:, 0]
        y = candidate_space[:, 1]
        z = acq_values

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=z, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label="Acquisition Function Value")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Acquisition Function Value")
        ax.set_title(f"Acquisition Function ({acquisition}) Over Search Space")
        plt.show()

    # Pairwise plots for higher-dimensional search spaces
    else:
        num_dimensions = candidate_space.shape[1]
        pairs = list(combinations(range(num_dimensions), 2))  # Generate all pairs of dimensions

        for dim1, dim2 in pairs:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                candidate_space[:, dim1], candidate_space[:, dim2],
                c=acq_values, cmap="viridis", alpha=0.7, s=200
            )

            plt.colorbar(scatter, label="Acquisition Function Value")
            plt.xlabel(f"Dimension {dim1}")
            plt.ylabel(f"Dimension {dim2}")
            plt.title(f"Acquisition Function ({acquisition}) - Dims {dim1} vs {dim2}")
            plt.grid()
            plt.show()

def plot_acquisition_function_(self, candidate_space, acq_values, acquisition="UCB"):

    print(f"{candidate_space.shape[1]=}")
    # Plot for 1D search space
    if candidate_space.shape[1] == 1:
        print(f"{candidate_space.shape[1] == 1}")
        plt.figure(figsize=(10, 6))
        plt.plot(candidate_space, acq_values, label=f"{acquisition} values")
        plt.xlabel("Search Space")
        plt.ylabel("Acquisition Function Value")
        plt.title(f"Acquisition Function ({acquisition}) Over Search Space")
        plt.legend()
        plt.grid()
        plt.show()

    # Plot for 2D search space
    elif candidate_space.shape[1] == 2:
        print(f"{candidate_space.shape[1] == 2}")

        x = candidate_space[:, 0]
        y = candidate_space[:, 1]
        z = acq_values

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=z, cmap="viridis", label=f"{acquisition} values")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Acquisition Function Value")
        ax.set_title(f"Acquisition Function ({acquisition}) Over Search Space")
        plt.legend()
        plt.imsave("acquisition_function.png")
        plt.show()