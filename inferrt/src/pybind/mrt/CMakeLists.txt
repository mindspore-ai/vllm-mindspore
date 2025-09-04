check_debug_log_out()

# Add mrt_api pybind11 sub module
pybind11_add_module(_mrt_api NO_EXTRAS pybind11_api.cc)
target_link_libraries(_mrt_api PUBLIC inferrt)

# Add mrt_ir pybind11 sub module
pybind11_add_module(_mrt_ir NO_EXTRAS pybind11_ir.cc)
target_link_libraries(_mrt_ir PUBLIC inferrt)
