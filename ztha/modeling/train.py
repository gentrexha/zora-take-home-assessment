from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from ztha.config import CONFIG

MODELS_DIR = Path(CONFIG.artifacts_path)
PROCESSED_DATA_DIR = Path(CONFIG.data.raw_data_path).parent / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
