#!/usr/bin/python
import sys, os, subprocess
import settings
from compiler import builder

# ----------- program usage -----------

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
    print "\tcpuda.py file.cu.cc"
    print "\t\tcompile .cu.cc file (result from .cu conversion) to .ptx"
    print

# ---------- use settings ----------

# command used to compile .cpp file with access to cpuda cuda.h library
gpp_cpp_cmd_str = settings.GPP_CPP_CMD

# command used to check .cu correctness (whether the file is compilable or not)
gpp_cu_cmd_str = settings.GPP_CU_CMD

# command used to compile .cu.cc file to .ptx file (shared library)
gpp_ptx_cmd_str = settings.GPP_PTX_CMD

# location of cpuda library directory (cuda.h implementation)
lib_dir = settings.LIB_DIR

cpuda_dir = os.path.dirname(os.path.realpath(__file__)) # get cpuda directory path
lib_path = os.path.join(cpuda_dir, lib_dir) # apply to it library relative location

# ---------- specific file processing ----------

# Processes .cpp file - call g++ to compile it including cuda.h directory
def process_cpp(cpp_file):
    print "CpUDA: processing CPP file"
    raw_file_name = os.path.splitext(os.path.basename(cpp_file))[0]
    gpp_cmd = gpp_cpp_cmd_str.format(lib_path, raw_file_name, cpp_file)
    print "call:", gpp_cmd
    subprocess.call(gpp_cmd)

# Determines whether .cu file is compilable - call g++ previously adding minimal header
def check_cu_correctness(cu_file):
    file_basename = os.path.basename(cu_file)
    ptx_file_name = os.path.splitext(file_basename)[0] + ".ptx"
    cu_cc_file = file_basename + ".cc"
    code = builder.Builder().generate_to_check(cu_file)
    with open(cu_cc_file, "w") as output_file:
        output_file.write(code)
    return subprocess.call(gpp_cu_cmd_str.format(ptx_file_name, cu_cc_file)) == 0

# Processes .cu file - call builder to generate .cu.cc file than process it
def process_cu(cu_file):
    print "CpUDA: processing CU file"
    print "check .cu correctness:"
    if check_cu_correctness(cu_file):
        print "ok"
        print "generate .cu.cc file with builder:",
        code = builder.Builder().generate_ptx_code(cu_file)
        cu_cc_file = os.path.basename(cu_file) + ".cc" # get file name (without directories) and add .cc extension
        with open(cu_cc_file, "w") as output_file:
            output_file.write(code)
        print "result saved in " + cu_cc_file
        print
        process_cu_cc(cu_cc_file)
    else:
        print "file is not valid, correct errors"

# Processes .cu.cc file - call g++ to compile it to .ptx file (shared library)
def process_cu_cc(cu_cc_file):
    print "CpUDA: processing CU.CC file"
    raw_file_name = os.path.splitext(os.path.basename(cu_cc_file))[0] # remove .cc and directories prefix
    ptx_file_name = os.path.splitext(raw_file_name)[0] + ".ptx" # then remove .cu and add .ptx extension
    gpp_ptx_cmd = gpp_ptx_cmd_str.format(ptx_file_name, cu_cc_file)
    print "call:", gpp_ptx_cmd
    subprocess.call(gpp_ptx_cmd)

# ---------- general file processing ----------

# Check file extension and call appropriate function
def process_file(in_file):
    (file_name, ext) = os.path.splitext(in_file)
    if ext == ".cpp":
        process_cpp(in_file)
    elif ext == ".cu":
        process_cu(in_file)
    elif ext == ".cc" and os.path.splitext(file_name)[1] == ".cu":
        process_cu_cc(in_file)
    else:
        print "Bad extension:", ext
        print "CpUDA only processes .cpp, .cu and .cu.cc files."

# ----------- program start -----------

if __name__ == "__main__":
    # interpreting command line
    if (len(sys.argv) == 1):
        print_usage()
    elif (len(sys.argv) == 2):
        process_file(sys.argv[1])
    else:
        print "Provide exactly one file."
