from ultralytics import YOLO

from ultralytics.utils.benchmarks import benchmark

# Load the Triton Server model
model = YOLO(f'http://localhost:8000/yolo', task='detect')

# Run inference on the server
results = model('/home/mirudhu/Documents/2024-02-20_17_53_04_119.jpg.4n4qoddt.ingestion-684cb69778-qb2ft.jpg')

# Benchmark on GPU
benchmark(model='/home/mirudhu/Documents/Counting/CVAT/CVAT_New/ep_m_model.pt', data='/home/mirudhu/Documents/Counting/CVAT/CVAT_New/detect/detect.yaml', imgsz=640, half=False, device=0)
