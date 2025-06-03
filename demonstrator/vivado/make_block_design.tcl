set overlay_name "block_design"
set design_name "radioml_demo"
set iprepo_dir {./../generate_transmitter/ip ./../generate_receiver/ip ./../generate_inspector/ip ./../generate_FINNadapter/ip ./finn_ip}

# Create project
create_project ${overlay_name} ./${overlay_name} -part xczu28dr-ffvg1517-2-e
set_property target_language Verilog [current_project]

# Set IP repository paths
set_property ip_repo_paths $iprepo_dir [current_project]
update_ip_catalog

# Add constraints
add_files -fileset constrs_1 -norecurse ./constraints.xdc

# Make block design
source ./${design_name}.tcl
