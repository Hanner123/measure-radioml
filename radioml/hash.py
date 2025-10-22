import hashlib
import argparse
import os

def calculate_file_hash(file_path):
    """Berechnet den SHA256-Hash einer Datei."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Berechne den SHA256-Hash einer Datei (.onnx oder .npz).")
    parser.add_argument("file", help="Pfad zur Datei (.onnx oder .npz)")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print("Datei existiert nicht.")
        return

    hash_value = calculate_file_hash(args.file)
    print(f"SHA256-Hash von {args.file}:\n{hash_value}")

if __name__ == "__main__":
    main()
