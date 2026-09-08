import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Load GRF data ---
df = pd.read_csv("grf_1.csv")

GRF_labels = [
    "ground_force_right_vx",
    "ground_force_right_vy",
    "ground_force_right_vz",
    "ground_force_left_vx",
    "ground_force_left_vy",
    "ground_force_left_vz",
]

time = df["time"].values
n_samples = len(df)

# --- Load video ---
cap = cv2.VideoCapture("Suhasno_1.mov")
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Resample each GRF component to video frames
grf_resampled = {}
for label in GRF_labels:
    grf_resampled[label] = np.interp(
        np.linspace(0, n_samples - 1, frame_count),
        np.arange(n_samples),
        df[label].values,
    )

# --- Video writer ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    "output_with_grf_6axis.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,  # slow motion ~50ms per frame
    (width * 2, height),
)

# --- Loop frames ---
window = int(fps * 2)  # 2 second sliding window
for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # --- Generate GRF subplot ---
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))  # 16:9-ish ratio
    fig.suptitle(
        "Ground Reaction Forces (Aligned with Video)", fontsize=14, fontweight="bold"
    )

    for i, label in enumerate(GRF_labels):
        ax = axs.flat[i]
        y = grf_resampled[label]

        start = max(0, frame_idx - window)
        ax.plot(
            np.arange(start, frame_idx + 1) / fps, y[start : frame_idx + 1], color="red"
        )

        ax.set_xlim((frame_idx - window) / fps, (frame_idx + 1) / fps)
        ax.set_ylim(min(y), max(y))
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Convert plot to numpy image
    canvas = FigureCanvas(fig)
    canvas.draw()
    plot_img = np.asarray(canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
    plt.close(fig)

    # Resize with smooth interpolation to avoid "stretchy text"
    plot_img = cv2.resize(plot_img, (width, height), interpolation=cv2.INTER_AREA)

    # Combine video + plot
    combined = np.hstack((frame, plot_img))
    out.write(combined)

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Saved as output_with_grf_6axis.mp4")
