"""
Generate the sampling window visualization for the thesis.
This creates a figure showing how the 10-second sampling window works.
"""
import sys
sys.path.insert(0, r"C:\Users\edgar\Documents\Studium\Mathe\Bachelor\Code")

import numpy as np
from codebase.data import SegmentToken
from codebase.miscellaneous.create_visuals import plot_sampling_window

# Create synthetic token data spanning 14 seconds
# These represent the segment endpoints from the full track
tokens = [
    SegmentToken(height=0.0, amount=0.5, time=0.0),    # Start at 0
    SegmentToken(height=0.8, amount=0.3, time=1.5),    # Rise
    SegmentToken(height=0.7, amount=0.6, time=3.0),    # Slight drop (crosses left boundary)
    SegmentToken(height=0.2, amount=0.8, time=5.0),    # Inside window
    SegmentToken(height=0.9, amount=0.4, time=7.0),    # Inside window
    SegmentToken(height=0.85, amount=0.5, time=9.5),   # Inside window
    SegmentToken(height=0.3, amount=0.7, time=11.0),   # Inside window (crosses right boundary)
    SegmentToken(height=0.6, amount=0.4, time=13.0),   # Outside window
    SegmentToken(height=0.4, amount=0.5, time=14.5),   # End
]

# Generate the figure
fig = plot_sampling_window(
    tokens=tokens,
    window_start=2.0,
    window_size=10.0,
    plot_start=0.0,
    plot_end=14.0
)

# Show interactive version
fig.show()
