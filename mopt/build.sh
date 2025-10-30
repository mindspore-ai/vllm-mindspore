#!/bin/bash
set -e

# Clean previous builds
rm -rf build dist

# Build the wheel
python -m build --wheel --no-isolation
