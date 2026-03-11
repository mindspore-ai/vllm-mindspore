add_subdirectory(pybind)

# Install Python files in ms_inferrt folder
install(DIRECTORY ms_inferrt/
        DESTINATION .)
        # FILES_MATCHING PATTERN "*.py"
        # PATTERN "__pycache__" EXCLUDE
        # PATTERN "*.pyc" EXCLUDE)
