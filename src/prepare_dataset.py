import os
import argparse
import shutil

from sklearn.model_selection import train_test_split


def process_class(class_name, source_dir, target_dir, seed=42):
    # class_dir = os.path.join(source_dir, class_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    files = os.listdir(source_dir)
    train_files, test_files = train_test_split(files, test_size=0.3, random_state=seed)
    val_files, test_files = train_test_split(
        test_files, test_size=0.66, random_state=seed
    )
    # Create folders if missing
    for folder in ["train", "val", "test"]:
        folder_path = os.path.join(target_dir, folder, class_name)
        os.makedirs(folder_path, exist_ok=True)
    # Move files
    for file in train_files:
        shutil.copyfile(
            os.path.join(source_dir, file),
            os.path.join(target_dir, "train", class_name, file),
        )
    for file in val_files:
        shutil.copyfile(
            os.path.join(source_dir, file),
            os.path.join(target_dir, "val", class_name, file),
        )
    for file in test_files:
        shutil.copyfile(
            os.path.join(source_dir, file),
            os.path.join(target_dir, "test", class_name, file),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the processed dataset.",
    )
    parser.add_argument("--classes_mapping", type=str, help="Classes maping")

    args = parser.parse_args()

    input_dir = args.input_dir.strip()
    output_dir = args.output_dir.strip()
    classes = args.classes_mapping.split("--")
    classes = [x.replace("'", "").strip() for x in classes if x.strip()]
    for c in classes:
        folder, class_name = c.split("^")
        folder = folder.strip()
        class_name = class_name.strip()
        print(f"Processing {class_name} from {folder}")
        process_class(
            class_name=class_name,
            source_dir=os.path.join(input_dir, folder),
            target_dir=output_dir,
        )
