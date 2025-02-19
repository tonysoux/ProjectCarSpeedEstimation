import kagglehub

# Download latest version
path = kagglehub.model_download("amitkumargurjar/yolov11-for-car-detection/pyTorch/default")

print("Path to model files:", path)