"""Fine-tune YOLO detector on traffic/parking boxes (Ultralytics API)."""
from __future__ import annotations
import typer
app = typer.Typer()

@app.command()
def main(data_yaml: str = "configs/detector.yaml", epochs: int = 50,
         model: str = "yolo11m.pt", out: str = "models/detector"):
    from ultralytics import YOLO
    m = YOLO(model)
    m.train(data=data_yaml, epochs=epochs, imgsz=640, project=out)
    print(f"done -> {out}; export: yolo export model={out}/weights/best.pt format=onnx")

if __name__ == "__main__":
    app()
