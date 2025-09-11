import os
import shutil
import fnmatch
from setuptools import setup
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    """Custom distribution class to indicate binary extensions exist"""
    def has_ext_modules(self):
        return True

def copy_so_files(src_dir, dst_root_dir, dst_lib_dir, special_so_patterns):
    """
    Copy .so files with special handling for files matching specific patterns
    
    Args:
        src_dir: Source directory containing .so files
        dst_root_dir: Destination root directory for special .so files
        dst_lib_dir: Destination directory for other .so files
        special_so_patterns: List of patterns for .so files that should go to root dir
    """
    os.makedirs(dst_root_dir, exist_ok=True)
    os.makedirs(dst_lib_dir, exist_ok=True)
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.so'):
                src_path = os.path.join(root, file)
                # Check if file matches any of the special patterns
                if any(fnmatch.fnmatch(file, pattern) for pattern in special_so_patterns):
                    # Copy matching .so files to root directory
                    shutil.copy2(src_path, os.path.join(dst_root_dir, file))
                else:
                    # Copy other .so files to lib directory
                    shutil.copy2(src_path, os.path.join(dst_lib_dir, file))

def copy_py_files(src_python_dir, dst_root_dir):
    """
    Copy Python files while preserving subdirectory structure
    """
    for root, _, files in os.walk(src_python_dir):
        for file in files:
            if file.endswith('.py'):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_python_dir)
                dst_path = os.path.join(dst_root_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure paths
src_dir = script_dir + '/build/inferrt/src'
python_src_dir = script_dir + '/inferrt/python/mrt'
temp_dir = os.path.join(script_dir, 'temp_build')
package_name = 'mrt'

# Define patterns for special .so files (supports wildcards)
special_so_patterns = [
    '_mrt_api*.so',    # Matches all .so files starting with _mrt_api
    '_mrt_ir*.so',     # Matches all .so files starting with _mrt_ir
    '_mrt_torch*.so'   # Matches all .so files starting with _mrt_torch
]

# Clean and create temporary directory structure
package_dir = os.path.join(temp_dir, package_name)
shutil.rmtree(temp_dir, ignore_errors=True)
os.makedirs(package_dir, exist_ok=True)

# Copy files to temporary directory
copy_so_files(
    src_dir=src_dir,
    dst_root_dir=package_dir,  # Destination for special .so files
    dst_lib_dir=os.path.join(package_dir, 'lib'),  # Destination for other .so files
    special_so_patterns=special_so_patterns
)
copy_py_files(python_src_dir, package_dir)

# Generate wheel package
setup(
    name=package_name,
    version='1.0',
    packages=[package_name],
    package_dir={'': 'temp_build'},
    package_data={
        package_name: [
            '*.so',       # Include .so files in root directory
            'lib/*.so',   # Include .so files in lib directory
            '**/*.py',    # Include all Python files recursively
        ],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    zip_safe=False,
)

# Move generated wheel to output directory and clean up
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
for whl in os.listdir('dist'):
    shutil.move(os.path.join('dist', whl), output_dir)
shutil.rmtree(temp_dir)  # Remove temporary build directory
shutil.rmtree('dist')    # Remove dist directory
print("Build completed. Output files:", os.listdir(output_dir))
