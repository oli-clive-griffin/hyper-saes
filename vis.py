import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go  # type: ignore

AXIS = dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                showspikes=False,
                range=[-1, 1],
    )

def visualize_weights_3d(
    lines: np.ndarray,
    # features_per_color: Optional[int] = None,
    # colors: Optional[list[str]] = None,
    # width: int = 800,
    # height: int = 800,
    # line_width: float = 4.0,
    # marker_size: float = 8.0,
) -> go.Figure:
    """
    Visualize 3D vectors as lines from the origin using Plotly.

    Args:
        lines: numpy array of shape (3, N) where each column is a 3D vector
        features_per_color: number of features to assign to each color. If None,
                          all features will be the same color
        colors: list of colors to cycle through. Defaults to ['#FFD700', '#32CD32']
                (yellow and green)
        width: width of the plot in pixels
        height: height of the plot in pixels
        line_width: width of the plotted lines
        marker_size: size of the markers at vector endpoints

    Returns:
        plotly Figure object
    """
    if not isinstance(lines, np.ndarray) or lines.shape[0] != 3:
        raise ValueError("lines must be a numpy array of shape (3, N)")

    # Create figure
    fig = go.Figure()

    # Add origin point
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=8, color="black"), name="Origin"))

    # Draw vectors
    num_vectors = lines.shape[1]

    for i in range(num_vectors):
        x, y, z = lines[:, i]

        # Create line from origin to endpoint
        fig.add_trace(
            go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0, z],
                mode="lines+markers",
                line=dict(color="green", width=2),
                marker=dict(size=8, color="green"),
                name=f"Vector {i+1}",
            )
        )

    # Configure layout
    fig.update_layout(
        width=800,
        height=800,
        showlegend=True,
        scene=dict(
            xaxis=AXIS,
            yaxis=AXIS,
            zaxis=AXIS,
            aspectmode="cube",  # Force equal aspect ratio
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),

        ),
    )

    fig.show()

def visualize_weights_2d(lines: np.ndarray):
    if lines.shape[0] != 2:
        raise ValueError("lines must be a numpy array of shape (2, N)")

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot origin
    ax.plot(0, 0, "k.", markersize=10)

    # Draw vectors
    num_vectors = lines.shape[1]

    for i in range(num_vectors):
        x, y = lines[:, i]
        ax.plot([0, x], [0, y], "-", color="green", linewidth=2, marker="o", markersize=8)

    # Configure plot
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    ax.axvline(x=0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.show()
    plt.close()


# Example usage
if __name__ == "__main__":
    # Create some example 3D vectors
    vectors = np.array(
        [
            [1.0, -0.5, 0.0, 0.8],  # x coordinates
            [0.5, 1.0, -1.0, 0.3],  # y coordinates
            [0.7, -0.3, 0.5, -0.6],  # z coordinates
        ]
    )

    visualize_weights_3d(vectors)
