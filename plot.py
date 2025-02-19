import ast
import importlib.metadata
import sys

def extract_imports(file_path):
    """Trích xuất danh sách thư viện được import từ file Python"""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    imported_modules = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split('.')[0])

    return imported_modules

def filter_installed_packages(packages):
    """Chỉ lấy các package được cài đặt bằng pip"""
    installed_packages = {pkg.metadata["Name"].lower(): pkg.version for pkg in importlib.metadata.distributions()}
    
    # Lọc bỏ các module thuộc Python chuẩn
    std_libs = set(sys.builtin_module_names)  # Lấy danh sách thư viện chuẩn Python
    filtered_packages = {pkg: installed_packages[pkg.lower()] for pkg in packages if pkg.lower() in installed_packages and pkg not in std_libs}
    
    return filtered_packages

# Đọc imports từ file main.py
file_path = "main.py"
imported_packages = extract_imports(file_path)

# Lọc các thư viện thực sự được cài bằng pip
package_versions = filter_installed_packages(imported_packages)

# Ghi vào requirements.txt
with open("requirements2.txt", "w") as f:
    for pkg, version in package_versions.items():
        f.write(f"{pkg}=={version}\n")

print("✅ Đã tạo file requirements.txt từ main.py!")
