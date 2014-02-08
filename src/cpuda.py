#!/usr/bin/python
import sys, os, subprocess
import settings
from compiler import builder

# ----------- program usage -----------

def print_usage():
    print
    print "CpUDA usage:"
    print
    print "\tcpuda.py file.cpp [file2.cpp...] [-c]"
    print "\t\tcompile and link .cpp files including cuda.h library version"
    print "\t  -c\tonly compile to object files (including cuda.h)"
    print
    print "\tcpuda.py file.o [file2.o...]"
    print "\t\tlink .o files into executable (naming in by first file)"
    print
    print "\tcpuda.py file.cu"
    print "\t\tconvert and compile .cu file to work on cpu"
    print
    print "\tcpuda.py file.cu.cc"
    print "\t\tcompile .cu.cc file (result from .cu conversion) to .ptx"
    print

# ---------- use settings ----------

# command used to compile .cpp files to object files with access to cpuda cuda.h library
gpp_cpp_cmd_str = settings.GPP_CPP_CMD

# command used to linked .o files into executable file with options to make cuda.h work
gpp_link_cmd_str = settings.GPP_LINK_CMD

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
def process_cpp(cpp_files, link):
    print "CpUDA: processing CPP files"
    print "compile to object files"
    gpp_cmd = gpp_cpp_cmd_str.format(LIB=lib_path, ARGS=' '.join(cpp_files))
    print "call:", gpp_cmd
    if subprocess.call(gpp_cmd, shell=True) != 0:
        print "error during compilation, correct errors"
        return
    if link:
        o_files = []
        for cpp_f in cpp_files:
            o_files.append(cpp_f[:-4] + ".o")
        print
        process_o(o_files)

def process_o(o_files):
    print "CpUDA: link .o files to executable"
    raw_file_name = os.path.splitext(os.path.basename(o_files[0]))[0]
    link_cmd = gpp_link_cmd_str.format(NAME=raw_file_name, ARGS=' '.join(o_files))
    print "call:", link_cmd
    subprocess.call(link_cmd, shell=True)

# Determines whether .cu file is compilable - call g++ previously adding minimal header
def check_cu_correctness(cu_file):
    file_basename = os.path.basename(cu_file)
    ptx_file_name = os.path.splitext(file_basename)[0] + ".ptx"
    cu_cc_file = file_basename + ".cc"
    code = builder.Builder().generate_to_check(cu_file)
    with open(cu_cc_file, "w") as output_file:
        output_file.write(code)
    return subprocess.call(gpp_cu_cmd_str.format(NAME=ptx_file_name, ARG=cu_cc_file)) == 0

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
    gpp_ptx_cmd = gpp_ptx_cmd_str.format(NAME=ptx_file_name, ARG=cu_cc_file)
    print "call:", gpp_ptx_cmd
    subprocess.call(gpp_ptx_cmd)

# ----------- program start -----------

if __name__ == "__main__":
    # interpreting command line
    if (len(sys.argv) == 1):
        print_usage()
    else:
        # Check file(s) extension and call appropriate function
        ext = os.path.splitext(sys.argv[1])[1] # first file extension specifies behaviour of program

        if ext == ".cpp":
            args = sys.argv[1:]
            full_compile = True
            if args[-1] == "-c":
                full_compile = False
                args.pop()
            for f_in in args:
                if os.path.splitext(f_in)[1] != ".cpp":
                    print "Provide list of files only with .cpp extension"
                    break;
            else:
                process_cpp(args, full_compile)
        elif ext == ".o":
            args = sys.argv[1:]
            for f_in in args:
                if os.path.splitext(f_in)[1] != ".o":
                    print "Provide list of files only with .o extension"
                    break;
            else:
                process_o(args)
        elif ext == ".cu":
            if len(sys.argv) > 2:
                print "You can provide only one .cu file."
            else:
                process_cu(sys.argv[1])
        elif ext == ".cc" and os.path.splitext(sys.argv[1][:-3])[1] == ".cu":
            if len(sys.argv) > 2:
                print "You can provide only one .cu.cc file."
            else:
                process_cu_cc(sys.argv[1])
        else:
            print "Bad extension:", ext
            print "CpUDA only processes .cpp, .cu and .cu.cc files."
