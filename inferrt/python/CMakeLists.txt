add_subdirectory(pybind)

# Install Python files in mrt folder
install(DIRECTORY mrt/
        DESTINATION .)
        # FILES_MATCHING PATTERN "*.py"
        # PATTERN "__pycache__" EXCLUDE
        # PATTERN "*.pyc" EXCLUDE)
