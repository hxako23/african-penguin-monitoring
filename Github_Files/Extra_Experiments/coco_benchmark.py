import glob
import time
import csv
from ultralytics import YOLO

COCO_IMG_DIR = "/mnt/LinuxBackup2/Datasets_of_interest/benchmark_datasets/sama-coco/valid/images"
WARMUP_FRAMES = 10

coco_images = glob.glob(COCO_IMG_DIR + "/*.jpg")
print(f"Found {len(coco_images)} COCO validation images")

MODELS = {
    "YOLOv11n": "/home/hluma/hluma_results/runs/yolov11n-3/weights/best.pt",
    "YOLOv11m": "/home/hluma/hluma_results/runs/yolov11m/weights/best.pt",
    "YOLOv26m": "/home/hluma/hluma_results/runs/yolo26m-2/weights/best.pt",
    "YOLOv26n": "/home/hluma/hluma_results/runs/yolov26n/weights/best.pt",
}

coco_results = []
for model_name, weights_path in MODELS.items():
    print(f"\nBenchmarking {model_name} on COCO validation images (speed only)...")
    model = YOLO(weights_path)

    for image_path in coco_images[:WARMUP_FRAMES]:
        model(image_path, imgsz=640, device="cpu", verbose=False) #test 640x640 images on the cpu

    per_image_times = []
    for image_path in coco_images:
        start = time.perf_counter()
        model(image_path, imgsz=640, device="cpu", verbose=False)
        per_image_times.append(time.perf_counter() - start)

    avg_latency_ms = (sum(per_image_times) / len(per_image_times)) * 1000
    fps = 1000 / avg_latency_ms
    print(f"  {model_name} on COCO images: {avg_latency_ms:.2f} ms/image, {fps:.2f} FPS")

    coco_results.append({
        "model": model_name,
        "dataset": "sama-coco-valid",
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 2),
        "n_images": len(per_image_times),
    })

with open("/home/hluma/hluma_results/coco_speed_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "dataset", "avg_latency_ms", "fps", "n_images"])
    writer.writeheader()
    for row in coco_results:
        writer.writerow(row)
print("\nSaved: coco_speed_benchmark.csv")
