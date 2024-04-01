from pathlib import Path

# Define paths
triton_repo_path = Path('tmp') / 'triton_repo'
triton_model_path = triton_repo_path / 'yolo'

# Create directories
(triton_model_path / '1').mkdir(parents=True, exist_ok=True)

# Move ONNX model to Triton Model path
onnx_file = "/home/mirudhu/Documents/Counting/CVAT/CVAT_New/ep_m_model.onnx"
Path(onnx_file).rename(triton_model_path / '1' / 'model.onnx')

# Create config file
(triton_model_path / 'config.pbtxt').touch()