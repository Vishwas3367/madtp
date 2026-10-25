import os
import glob

def find_orphaned_files():
    # 1. Define your main entry points (files you run directly)
    entry_points = {"live_madtp.py", "api_server.py", "dashboard/dashboard.py"}
    
    # 2. Get all Python files in the project
    all_py_files = glob.glob("**/*.py", recursive=True)
    all_py_files = [f for f in all_py_files if "venv" not in f and "__pycache__" not in f]
    
    # 3. Read the content of all files to check for imports
    file_contents = {}
    for file_path in all_py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            file_contents[file_path] = f.read()

    print("=" * 50)
    print(" ORPHANED FILE SCANNER")
    print("=" * 50)

    orphans = []
    
    for target_file in all_py_files:
        # Skip entry points and this script itself
        if any(target_file.endswith(ep) for ep in entry_points) or target_file == "find_orphans.py":
            continue
            
        # Get the module name (e.g., 'madtp/risk_prediction.py' -> 'risk_prediction')
        module_name = os.path.basename(target_file).replace(".py", "")
        
        is_imported = False
        for search_file, content in file_contents.items():
            if target_file == search_file:
                continue
                
            # Check if the module name is imported in any other file
            if f"import {module_name}" in content or f"from {module_name}" in content or f"from .{module_name}" in content or module_name in content:
                is_imported = True
                break
                
        if not is_imported:
            orphans.append(target_file)

    if not orphans:
        print("✓ No orphaned files found. Your repository is clean.")
    else:
        print(f"⚠ Found {len(orphans)} files that are NEVER imported by your system:\n")
        for orphan in sorted(orphans):
            print(f"  - {orphan}")
            
    print("\nNote: Always double-check this list before deleting. Some scripts might be standalone utilities.")

if __name__ == "__main__":
    find_orphaned_files()