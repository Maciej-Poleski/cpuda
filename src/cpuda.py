#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import subprocess

# ----------- Program usage -----------

def print_usage():
    print
    print "CpUDA usage:"
    print
    print "\tcpuda.py file.cpp"
    print "\t\tcompile .cpp file including cpuda cuda.h library version"
    print
    print "\tcpuda.py file.cu"
    print "\t\tconvert and compile .cu file to work on cpu"
    print

# ---------- File processing ----------

gpp_cmd_str = "g++ -std=c++0x -I {0} -o {1} {2} -Wl,--stack,100000" # in order to use provide: cuda.h directory, output name, file to compile
lib_dir = "library" # name of directory with cuda.h library (relative to cpuda directory)

cpuda_dir = os.path.dirname(os.path.realpath(__file__)) # load cpuda directory path

# Processes .cpp file - call g++ to compile it including cuda.h directory
def process_cpp(cpp_file):
    print "CpUDA: processing CPP file"
    lib_path = os.path.join(cpuda_dir, lib_dir)
    raw_file_name = os.path.splitext(os.path.basename(cpp_file))[0]
    gpp_cmd = gpp_cmd_str.format(lib_path, raw_file_name, cpp_file)
    print "call:", gpp_cmd
    subprocess.call(gpp_cmd)

# Processes .cu file
def process_cu(cu_file):
    print "CpUDA: processing CU file"
    proc=subprocess.Popen('python builder.py {0}'.format(os.path.abspath(cu_file)),shell=True,cwd='compiler')
    return proc.wait()

# Check file extension and call appropriate function
def process_file(in_file):
    ext = os.path.splitext(in_file)[1]
    if ext == ".cpp":
        process_cpp(in_file)
    elif ext == ".cu":
        process_cu(in_file)
    else:
        print "Bad extension:", ext
        print "CpUDA only processes .cpp and .cu files."

# ----------- Program start -----------

# interpreting command line
if (len(sys.argv) == 1):
    print_usage()
elif (len(sys.argv) == 2):
    process_file(sys.argv[1])
else:
    print "Provide exactly one file."
