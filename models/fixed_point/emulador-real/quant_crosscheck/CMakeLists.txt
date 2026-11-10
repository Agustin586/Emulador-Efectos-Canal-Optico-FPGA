################################################################################
# ██████████████████████████████████████████████████████████████████████████████
# █▀▀▀▀███▀▀▀▀█████▀▀▀▀▀██████▀▀▀█████████▀▀▀▀▀▀▀███████▀▀▀▀▀▀▀█████▀▀▀▀████▀▀▀█
# █    ███    ████▌      ████▌   ████████    ▄▄    ███▀   ▄▄▄   ▀███     ▀██   █
# █    ▀▀▀    ████   ▄   ▐███▌   ███████    ████▄▄▄██▌   ▐███▌   ▐██       ▀   █
# █           ███   ▐█▌   ███▌   ███████    █████████▌   ▐███▌    ██   ▄       █
# █    ███    ██▌          ██▌   ███████▄   ▀██▀   ███    ███    ███   ██▄     █
# █    ███    ██   █████   ██▌        ████▄      ▄█████▄       ▄████   ████▄   █
# ██████████████████████████████████████████████████████████████████████████████
# █████████████████████████ DSP SIMULATION ENGINE ██████████████████████████████
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
################################################################################
# Author: Patricio Reus Merlo
# Date: 05/21/2024
################################################################################
# MIT License
# 
# Copyright (c) 2024 Fundacion Fulgor
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

################################################################################
# USER SETTINGS (BEGIN)
################################################################################

#######################################
# APP SETTINGS
#######################################

# Application main file name
if(NOT DEFINED TARGET_FILE)
    set(TARGET_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)
endif()

# Executable file name
if(NOT DEFINED EXE_NAME)
    set(EXE_NAME simulator)
endif()

#######################################
# MODULES STATIC LIBRARY
#######################################

# Base directory
if(NOT DEFINED EXTERNAL_DIR)
    set(EXTERNAL_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/modules/)
endif()

# Modules
if(NOT DEFINED EXTERNAL_LIST)
    set(EXTERNAL_LIST

    )
endif()

#######################################
# LOCAL STATIC LIBRARY
#######################################

# Base directory
if(NOT DEFINED LOCAL_DIR)
    set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR}/src/)
endif()

# Modules
if(NOT DEFINED LOCAL_LIST)
    set(LOCAL_LIST
        root
    )
endif()

#######################################
# HALCON CORE (RELEASE)
#######################################

if(NOT DEFINED CORE_DIR)
    set(CORE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/core/)
endif()

if(NOT DEFINED CORE_TYPE)
    set(CORE_TYPE "FULL")
endif()

#######################################
# RUN PATH
#######################################

if(NOT DEFINED INSTALL_DIR)
    set(INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/run/)
endif()

################################################################################
# USER SETTINGS (END)
################################################################################

################################################################################
# PROJECT COMPILATION (BEGIN)
################################################################################

#######################################
# HALCON BANNER
#######################################

set(BANNER
"
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
██ SIMULATOR POWERED BY ... ████████████████████████████████████████████████████
████████████████████████████████████████████████████████████████████████████████
██▀▀▀█████▀▀▀███████▀▀▀███████▀▀███████████▀▀▀▀▀▀████████▀▀▀▀██████▀▀▀█████▀▀▀██
██░░░░███▌░░░█████▌░░░░░█████░░░░████████▀ ░░░░░░░▀███▀░░░▄▄░░░▀██▌░░░░▀███░░░██
██░░░░███▒░░░█████░░▐░░░▐████░░░░███████ ░░░████▄▄▄██░░░░████░░░░█▌░░░░░ ▀█░░░██
██░░░░░░░░░░░████░░░██░░░▀███░░░░███████░░░░█████████░░░░████░░░░█▌░░░▄░░░▀░░░██
██░░░░███▒░░░███░░░░░░░░░░███░░░░███████ ░░░████▀▀▀██░░░░████░░░░█▌░░░██ ░░░░░██
██░░░░███▌░░░██▀░░▐████░░░░██░░░░░░░░░▒██▄░░░░░░░░▄███▄░░░▀▀░░░▄██▌░░░███▄░░░░██
██▄▄▄█████▄▄▄██▄▄▄██████▄▄▄██▄▄▄▄▄▄▄▄▄██████▄▄▄▄█████████▄▄▄▄██████▄▄▄█████▄▄▄██
██████████████████████████ DSP SIMULATION ENGINE ███████████████████████████████
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
"
)

message(${BANNER})

#######################################
# PROJECT NAME
#######################################

project(${EXE_NAME})

#######################################
# CMAKE AND GXX COMPILATION FLAGS
#######################################

# CMake version
cmake_minimum_required(VERSION 3.10)

# Use g++ > 13
set(CMAKE_CXX_COMPILER g++-13)

# Compilation type
set(CMAKE_BUILD_TYPE Release)

# Use the C++20 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# Enable all standard warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")

# Treat all warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

# Warn about comparisons between signed and unsigned values
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare")

# Disable misleading indentation warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-misleading-indentation")

# Enable vectorization of loops
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")

# Enable most warning messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Enable extra warning messages
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

# Issue all the warnings demanded by strict ISO C and ISO C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

# Warn whenever a local variable shadows another variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")

# Warn if testing floating point values for equality
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")

# Enable format string warnings (with extra checks)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=2")

# Warn for implicit type conversions that may alter a value
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")

# Warn about suspicious uses of logical operators
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op")

# Warn if an include directory does not exist
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-include-dirs")

# Warn if a null dereference is detected
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference")

# Warn if string literals are longer than the maximum length
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverlength-strings")

# Warn on anything that is unused
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused")

# Warn if a variable is used without being initialized
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")

# Warn if a variable might be used uninitialized
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")

# Warn if a switch statement does not cover all enum values
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-enum")

# Strict aliasing rules (level 2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing=2")

# Warn if a float is promoted to a double
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion")

# Warn about implicit fallthrough in switch statements
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimplicit-fallthrough")

# Warn if anything is declared more than once
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")

# Optimize for the host architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Enable stack protection
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")

# Enable additional compile-time and run-time checks
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2")

# Conditional flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")

    message("-- Build as Release")

    # Optimization level 3 (higher optimization)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

else()

    message("-- Build as Debug")

    # Optimization level 0 (lower optimization)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

    # Generate debug information
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

    # Enable address sanitizer (detects memory errors)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

    # Enable undefined behavior sanitizer
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")

    # Do not omit the frame pointer (useful for debugging)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

    # Disable inline expansion for easier debugging
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline")

endif()

#######################################
# CONDITIONAL COMPILATION MACROS
#######################################

if(CORE_TYPE STREQUAL "FULL")
    add_definitions(-D_RUN_POSEDGE_LOGIC_ONLY=0)
else()
    add_definitions(-D_RUN_POSEDGE_LOGIC_ONLY=1)
endif()

#######################################
# CORE PATH SETTINGS
#######################################

set(CORE_INC_DIR ${CORE_DIR}inc/)
set(CORE_BIN_DIR ${CORE_DIR}src/)

set(CORE_FULL_BIN lib_halcon_core.a)
set(CORE_POSEDGE_BIN lib_halcon_core_only_posedge.a)

set(CORE_DEPENDENCES
    yaml-cpp
    Boost::boost
    fftw3
)

find_package(Boost 1.71 REQUIRED)
find_library(fftw3 /usr/include/fftw3.h)

#######################################
# CORE VERSION SELECTION
#######################################

if(CORE_TYPE STREQUAL "FULL")
    set(CORE_LIB ${CORE_BIN_DIR}${CORE_FULL_BIN})
    message("-- Loading FULL HALCON core")
else()
    set(CORE_LIB ${CORE_BIN_DIR}${CORE_POSEDGE_BIN})
    message("-- Loading POSEDGE HALCON core")
endif()

#######################################
# EXTERNAL STATIC LIBRARY
#######################################

list(LENGTH EXTERNAL_LIST EXTERNAL_LEN)

if(${EXTERNAL_LEN} GREATER 0)

    foreach(MODULE ${EXTERNAL_LIST})

        message("-- Loading external module ${MODULE} from ${EXTERNAL_DIR}${MODULE}")

        # C++ support    
        file(GLOB TMP_SRC ${EXTERNAL_DIR}${MODULE}/*.cpp)
        list(APPEND MODULES_SRC ${TMP_SRC})

        file(GLOB TMP_HEADER ${EXTERNAL_DIR}${MODULE}/*.hpp)
        list(APPEND MODULES_HEADER ${TMP_HEADER})

        # C support
        file(GLOB TMP_SRC ${EXTERNAL_DIR}${MODULE}/*.c)
        list(APPEND MODULES_SRC ${TMP_SRC})

        file(GLOB TMP_HEADER ${EXTERNAL_DIR}${MODULE}/*.h)
        list(APPEND MODULES_HEADER ${TMP_HEADER})
        
        # Include directories
        list(APPEND MODULES_INCLUDE_DIRS ${EXTERNAL_DIR}${MODULE})

    endforeach()

    add_library(external STATIC ${MODULES_SRC} ${MODULES_HEADER})

    list(APPEND MODULES_INCLUDE_DIRS ${CORE_INC_DIR})

    target_include_directories(external PUBLIC ${MODULES_INCLUDE_DIRS})

endif()

#######################################
# LOCAL STATIC LIBRARY
#######################################

foreach(MODULE ${LOCAL_LIST})

    message("-- Loading local module ${MODULE} from ${LOCAL_DIR}${MODULE}")

    # C++ support    
    file(GLOB TMP_SRC ${LOCAL_DIR}${MODULE}/*.cpp)
    list(APPEND LOCAL_SRC ${TMP_SRC})

    file(GLOB TMP_HEADER ${LOCAL_DIR}${MODULE}/*.hpp)
    list(APPEND LOCAL_HEADER ${TMP_HEADER})

    # C support
    file(GLOB TMP_SRC ${LOCAL_DIR}${MODULE}/*.c)
    list(APPEND LOCAL_SRC ${TMP_SRC})

    file(GLOB TMP_HEADER ${LOCAL_DIR}${MODULE}/*.h)
    list(APPEND LOCAL_HEADER ${TMP_HEADER})

    # Include directories
    list(APPEND LOCAL_INCLUDE_DIRS ${LOCAL_DIR}${MODULE})

endforeach()

add_library(local STATIC ${LOCAL_SRC} ${LOCAL_HEADER})

list(APPEND LOCAL_INCLUDE_DIRS ${CORE_INC_DIR})

if(${EXTERNAL_LEN} GREATER 0)
    list(APPEND LOCAL_INCLUDE_DIRS ${MODULES_INCLUDE_DIRS})
endif()

target_include_directories(local PUBLIC ${LOCAL_INCLUDE_DIRS})

#######################################
# APPLICATION COMPILING
#######################################

add_executable(${EXE_NAME} ${TARGET_FILE})

#######################################
# APPLICATION LINKING
#######################################

set(EXE_LIBS
    local
    ${CORE_LIB}
    ${CORE_DEPENDENCES}
)

if(${EXTERNAL_LEN} GREATER 0)

    list(APPEND EXE_LIBS external)

endif()

set_target_properties(${EXE_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${EXE_NAME} PUBLIC ${EXE_LIBS})

#######################################
# APPLICATION INSTALLATION
#######################################

install(TARGETS ${EXE_NAME} DESTINATION ${INSTALL_DIR})

################################################################################
# PROJECT COMPILATION (END)
################################################################################