
from src.data import load_datasets

train_ds, val_ds, test_ds = load_datasets()

print("\nDataset loading successful!")
print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Test batches:", len(test_ds))

images, labels = next(iter(train_ds))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image pixel range:", float(images.numpy().min()),
      "to", float(images.numpy().max()))