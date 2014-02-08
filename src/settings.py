# ----------------------------------------------
#       This file contains cpuda settings.
# ----------------------------------------------

# command used to compile .cpp files to object files with access to cpuda cuda.h library
# cpuda provides cuda.h directory as LIB and files to compile as ARGS
GPP_CPP_CMD = "g++ -std=c++0x -I {LIB} -c {ARGS}"

# command used to linked .o files into executable file with options to make cuda.h work
# cpuda provides output name as NAME and files to link as ARGS
GPP_LINK_CMD = "g++ -Wl,--stack,100000 -o {NAME} {ARGS}"

# command used to check .cu correctness (whether the file is compilable or not)
# cpuda provides output name as NAME and file to compile as ARG
GPP_CU_CMD = "g++ -o {NAME} -c {ARG}"

# command used to compile .cu.cc file to .ptx file (shared library)
# cpuda provides output name as NAME and file to compile as ARG
GPP_PTX_CMD = "g++ -std=c++11 -O3 -march=native -shared -fPIC -xc++ -pipe -o {NAME} {ARG}"

# relative (to cpuda.py) location of cpuda library directory (cuda.h implementation)
# cpuda uses its location joined with this path in order to include cuda.h
LIB_DIR = "library"
