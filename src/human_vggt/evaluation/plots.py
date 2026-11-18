from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_correspondence(rows: list[dict], destination: Path) -> None:
    valid = [row for row in rows if row["status"] == "ok"]
    if not valid:
        return
    destination.parent.mkdir(exist_ok=True)
    fig = Figure(figsize=(7, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots()
    for site in dict.fromkeys(row["site"] for row in valid):
        points = [row for row in valid if row["site"] == site]
        axes.scatter(
            [r["angle_deg"] for r in points],
            [r["recall_at_1"] for r in points],
            label=site,
        )
    axes.set(
        xlabel="Camera baseline angle (degrees)",
        ylabel="Retrieval recall@1",
        ylim=(0, 1.05),
    )
    axes.legend()
    fig.savefig(destination, dpi=150)
