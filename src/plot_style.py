import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e6e5e0"

# Color follows the condition
WITHOUT_ROT = "#2a78d6"  # categorical slot 1
WITH_ROT = "#eb6834"     # categorical slot 2

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_PRIMARY,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
