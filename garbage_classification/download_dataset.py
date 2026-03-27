from pathlib import Path

import typer

from garbage_classification.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("asdasdasasdas/garbage-classification", output_dir=RAW_DATA_DIR, force_download=True)

    print("Path to dataset files:", path)

if __name__ == "__main__":
    app()
