from ultralytics import YOLO

# Load the Triton Server model
model = YOLO(f'http://localhost:8000/yolo', task='detect')

# Run inference on the server
results = model('/home/loki/2024-02-20_17_53_04_119.jpg.4n4qoddt.ingestion-684cb69778-qb2ft.jpg')