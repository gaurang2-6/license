import zipfile
import os

archive_path = r"C:\Users\gaura_18dowjl\.cache\kagglehub\datasets\saisirishan\indian-vehicle-dataset\1.archive"
base_abs = os.path.abspath(os.path.join("data", "indian_vehicle_dataset"))
dest_dir = "\\\\?\\" + base_abs

print("Opening zip archive...")
with zipfile.ZipFile(archive_path, 'r') as zf:
    names = zf.namelist()
    print(f"Total entries in archive: {len(names)}")
    print("Sample entries:")
    for n in names[:15]:
        print(" ", n)
    
    # Check if there are annotations / csv / xml / json
    ann_files = [n for n in names if any(n.endswith(ext) for ext in ('.csv', '.xml', '.json', '.txt'))]
    print(f"Annotation / text files ({len(ann_files)}):")
    for n in ann_files[:10]:
        print(" ", n)

    print(f"Extracting to: {dest_dir}...")
    os.makedirs(dest_dir, exist_ok=True)
    # Extract entries safely with extended path syntax
    count = 0
    for member in zf.infolist():
        target_path = os.path.join(dest_dir, member.filename.replace('/', os.sep))
        if member.is_dir():
            os.makedirs(target_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
            count += 1
            if count % 100 == 0 or count == len(names):
                print(f"Extracted {count}/{len(names)} files...")

print("Extraction complete!")
