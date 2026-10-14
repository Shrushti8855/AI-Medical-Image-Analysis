
from src.model import build_model

model = build_model()

print("\nModel created successfully!")
model.summary()

print("\nOutput shape:", model.output_shape)
print("Total parameters:", model.count_params())