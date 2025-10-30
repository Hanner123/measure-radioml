import numpy as np

data = np.load("context_output_jetson.npz")

# for key in npz.files:
#     arr = npz[key]
#     print(f"{key}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes/1024:.1f} KB")



# Die ersten 10 Keys nehmen (oder weniger, falls weniger vorhanden)
for i, key in enumerate(data.files[151:180]):
    arr = data[key]
    print(f"===== [{i+1}] {key} =====")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    print("Inhalt:")
    print(arr)  # komplettes Array ausgeben
    print("\n" + "-" * 80 + "\n")