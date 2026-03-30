import kagglehub

# Download latest version
path = kagglehub.dataset_download("romangolubeff/factoryplanning-tasks")

print("Path to dataset files:", path)