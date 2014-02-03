# ----------------------------------------------
#       This file contains cpuda settings.
# ----------------------------------------------

# command used to compile .cpp file with access to cpuda cuda.h library
# cpuda provides cuda.h directory, output name, file to compile
GPP_CPP_CMD = "g++ -std=c++0x -I {0} -o {1} {2} -Wl,--stack,100000"

# command used to check .cu correctness (whether the file is compilable or not)
# cpuda provides output name, file to compile
GPP_CU_CMD = "g++ -o {0} -c {1}"

# command used to compile .cu.cc file to .ptx file (shared library)
# cpuda provides output name, file to compile
GPP_PTX_CMD = "g++ -std=c++11 -O3 -march=native -shared -fPIC -xc++ -pipe -o {0} {1}"

# relative (to this file) location of cpuda library directory (cuda.h implementation)
# cpuda uses it's location joined with this path in order to include cuda.h
LIB_DIR = "library"
