import os
import random
import shutil
from collections import defaultdict

original_path = "UTKFace"
new_path = "UTKFace_small"

TARGET_SIZE = 5000   # change if needed

if not os.path.exists(new_path):
    os.makedirs(new_path)

# Group files by (gender, ethnicity)
groups = defaultdict(list)

for file in os.listdir(original_path):
    try:
        parts = file.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        ethnicity = int(parts[2])

        key = (gender, ethnicity)
        groups[key].append(file)

    except:
        continue

# Calculate how many per group
num_groups = len(groups)
per_group = TARGET_SIZE // num_groups

selected_files = []

for key in groups:
    selected = random.sample(groups[key], min(per_group, len(groups[key])))
    selected_files.extend(selected)

print("Selected images:", len(selected_files))

# Copy selected images
for file in selected_files:
    shutil.copy(
        os.path.join(original_path, file),
        os.path.join(new_path, file)
    )

print("Reduced dataset created at:", new_path)