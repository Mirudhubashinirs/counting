from ultralytics import YOLO

# Load a model
model = YOLO('/home/loki/ep_m_model.pt')  # load an official model

# Export the model
onnx_file = model.export(format='onnx', dynamic=True)
