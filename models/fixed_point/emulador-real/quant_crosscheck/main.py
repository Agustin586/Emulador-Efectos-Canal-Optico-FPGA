#!/usr/bin/python3
################################################################################
# Test cruzado de cuantización (paso C3.0 de GUIA_ISI_CD.md, decisión 33)
################################################################################
# Author: Wanda Molina
# Date: 11/10/2026
################################################################################
# Adaptado de tp2/main.py de HALCON (MIT License, Copyright (c) 2024 Fundacion
# Fulgor). Corre dentro del contenedor hawk_workspace a través de
# run_crosscheck.sh, que deja la biblioteca de HALCON en ../lib/.
################################################################################

################################################################################
# PYTHON MODULES
################################################################################

import os

################################################################################
# HALCON MODULES
################################################################################

from halcon import SimulatorHandler

################################################################################
# SIMULATOR HANDLER
################################################################################

simh = SimulatorHandler()

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

args = simh.get_command_line_args()

################################################################################
# SETTINGS: DIRECTORIES
################################################################################

simh.project_abs_dir = os.getcwd()
simh.build_dir = "build/"
simh.conf_dir = "conf/"
simh.run_dir = "run/"
simh.logs_dir = "run/logs/"
simh.src_dir = "src/"
simh.tests_dir = "tests/"
simh.modules_dir = "../lib/modules/"
simh.core_dir = "../lib/core/"

################################################################################
# SETTINGS: FILES
################################################################################

simh.cmakelists_file = "CMakeLists.txt"
simh.command_file = "conf/command.cmd"
simh.settings_file = "conf/settings.yaml"
simh.hierarchy_file = "conf/hierarchy.txt"
simh.main_cpp_file = "src/main.cpp"
simh.main_py_file = "main.py"
simh.binary_file = "run/simulator"

################################################################################
# SETTINGS: INIT
################################################################################

simh.init()

################################################################################
# SETTINGS: MODULES
################################################################################

if args.compile_f or args.compile_p:
    local_modules = simh.get_available_local_modules()
    simh.add_local_modules(local_modules)

################################################################################
# COMPILATION
################################################################################

if args.delete:
    simh.delete_old_files()

if args.enable_logs:
    simh.enable_compilation_logs()

if args.compile_f or args.compile_p:
    simh.set_make_parallelism(4)

if args.compile_f:
    simh.compile_full()

if args.compile_p:
    simh.compile_posedge()

################################################################################
# SETTINGS: LOG AND SET COMMANDS (no se usan: el simulador lee y escribe
# archivos en QC_DATA_DIR)
################################################################################

if args.run:
    simh.clean_command_file()

################################################################################
# SETTINGS: UPDATE YAML FILE
################################################################################

simh.update_settings_section()

# @ AUTO_SETTINGS_BEGIN

simh.settings['root']['enable_log'] = 0
simh.settings['root']['fs_hz'] = 1
simh.settings['root']['logger_buffer_size'] = 1000
simh.settings['root']['n_iterations'] = 1

# @ AUTO_SETTINGS_END

simh.save_settings_yaml()

################################################################################
# RUN SIMULATOR
################################################################################

if args.run:
    simh.run()
