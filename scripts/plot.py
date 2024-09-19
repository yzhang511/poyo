import matplotlib.pyplot as plt
import matplotlib.collections as collections
import numpy as np
from typing import Dict, Optional, Union


def plot_intervals_dict(
    intervals_dict: Dict[str, "Interval"],
    session_start: Optional[float] = None,
    session_end: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a dictionary of intervals for easy visualization of splits.

    Parameters:
    intervals_dict (dict): A dictionary where keys are labels and values are Interval objects.
    session_start (float, optional): The start time of the session. If None, it will be set to the minimum start time in the intervals_dict.
    session_end (float, optional): The end time of the session. If None, it will be set to the maximum end time in the intervals_dict.
    save_path (str, optional): The path where the plot should be saved. If None, the plot will be displayed using plt.show().

    Returns:
    None

    NOTE: This helper is not used anywhere in the code as of now, but it can be useful for visualizing the splits in the data, when debugging prepare_data files or exploring the prepared data.
    """
    if not intervals_dict:
        print("No intervals to plot.")
        return

    # Determine the plotting boundaries if not provided
    all_starts = np.concatenate(
        [interval.start for interval in intervals_dict.values()]
    )
    all_ends = np.concatenate([interval.end for interval in intervals_dict.values()])

    if session_start is None:
        session_start = all_starts.min()
    if session_end is None:
        session_end = all_ends.max()

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as necessary
    plt.subplots_adjust(left=0.3)  # Adjust left margin as necessary to fit labels
    plt.setp(
        ax.get_yticklabels(), rotation=45, ha="right"
    )  # Rotate labels for better fit
    labels = list(intervals_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    current_position = 0
    # Plot each collection of intervals with corresponding label and color
    for label, interval in intervals_dict.items():
        for start, end in zip(interval.start, interval.end):
            ax.broken_barh(
                [(start, end - start)],
                (current_position, 1),
                facecolors=colors[current_position % len(colors)],
            )
        current_position += 1

    ax.set_ylim(0, current_position)
    ax.set_xlim(session_start, session_end)
    ax.set_xlabel("Time")
    ax.set_yticks(np.arange(0.5, len(labels) + 0.5))
    ax.set_yticklabels(labels)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
