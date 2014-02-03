#!/usr/bin/env python3
from __future__ import print_function
import sys
import os
import subprocess

from cuparser import Parser

# ---------- builder class ----------

class Builder:
    class ObjectInfo(object):
        def __init__(self, name, type):
            self.name = name
            self.type = type

    class ParserCallbacks:
        def __init__(self):
            self.kernels = {}
            self.shared_objects = {}

        def report_kernel(self, name, args):
            self.kernels[name] = args

        def transform_shared_memory_definition(self, fun_name, var_name, var_type):
            if fun_name not in self.shared_objects:
                self.shared_objects[fun_name] = []
            self.shared_objects[fun_name] += [Builder.ObjectInfo(var_name, var_type)]
            return 'auto& {var_name} = detail::shared::{fun_name}[detail::block_flat_idx()]->{var_name};\n'.format(
                fun_name=fun_name,
                var_name=var_name)

    def generate_ptx_code(self, cu_file):
        callbacks = Builder.ParserCallbacks()
        parser = Parser()

        result = parser.parse(cu_file, callbacks)
        kernels = callbacks.kernels
        shared_objects = callbacks.shared_objects
        code = ""


        with open(os.path.dirname(os.path.realpath(__file__))+'/templates/header_constant.cc') as f:
            code += f.read()

        code += ('\nnamespace detail\n'
                 '{\n'
                 'namespace shared\n'
                 '{\n')
        for fun in shared_objects:
            code += 'struct {\n'
            for var in shared_objects[fun]:
                code += '\ttype_wrapper<{type}>::type {name};\n'.format(type=var.type, name=var.name)
            code += '}}** {};\n'.format(fun)

        code += ('}\n'
                 '};\n\n')

        code += '// ' + 20 * '-' + ' KERNEL CODE ' + 20 * '-' + '\n'
        code += result
        code += '// ' + 53 * '-' + '\n'

        code += ('\n#include <type_traits>\n'
                 '\n'
                 'extern "C" void _kernel_global_init(detail::gridDim_t gridDim, detail::blockDim_t blockDim, int throat_size)\n'
                 '{\n'
                 '\tdetail::THROAT_SIZE = throat_size;\n'
                 '\tdetail::BLOCK_SIZE = detail::grid_flat_size(blockDim);\n'
                 '\tdetail::blocks_synchronization = new detail::throat_sync*[detail::grid_flat_size(gridDim)];\n'
                 )

        for fun in shared_objects:
            code += ('\n'
                     '\tusing {name}_t = std::remove_pointer<decltype(detail::shared::{name})>::type;\n'
                     '\tdetail::shared::{name} = new {name}_t[detail::grid_flat_size(gridDim)];\n').format(name=fun)

        code += '}\n\n'

        code += ('extern "C" void _kernel_global_deinit()\n'
                 '{\n\tdelete [] detail::blocks_synchronization;\n')
        for fun in shared_objects:
            code += '\tdelete [] detail::shared::{name};\n'.format(name=fun)

        code += '}\n\n'

        code += ('extern "C" void _kernel_block_init(detail::gridDim_t gridDim, detail::blockIdx_t blockIdx)\n'
                 '{\n\tdetail::blocks_synchronization[detail::block_flat_idx(gridDim, blockIdx)] = new detail::throat_sync(detail::BLOCK_SIZE);\n')
        for fun in shared_objects:
            code += ('\tusing {name}_t = std::remove_pointer<decltype(detail::shared::{name})>::type;\n'
                     '\tdetail::shared::{name}[detail::block_flat_idx(gridDim, blockIdx)] = new std::remove_pointer<{name}_t>::type;\n'
            ).format(name=fun)
        code += '}\n\n'

        code += ('extern "C" void _kernel_block_deinit(detail::gridDim_t gridDim, detail::blockIdx_t blockIdx)\n'
                 '{\n\tdelete detail::blocks_synchronization[detail::block_flat_idx(gridDim, blockIdx)];\n')
        for fun in shared_objects:
            code += '\tdelete detail::shared::{name}[detail::block_flat_idx(gridDim, blockIdx)];\n'.format(name=fun)
        code += '}\n\n'

        for fun in kernels:
            code += ('extern "C" void {name}_start(\n\tvoid* args[],\n'
                     '\tdetail::gridDim_t gridDim,\n'
                     '\tdetail::blockDim_t blockDim,\n'
                     '\tdetail::blockIdx_t blockIdx,\n'
                     '\tdetail::threadIdx_t threadIdx)\n'
                     '{{\n'
                     '\t::gridDim = gridDim;\n'
                     '\t::blockDim = blockDim;\n'
                     '\t::blockIdx = blockIdx;\n'
                     '\t::threadIdx = threadIdx;\n'
                     '\tdetail::throat_sync &block_sync =  *detail::blocks_synchronization[detail::block_flat_idx(gridDim, blockIdx)];\n'
                     '\tblock_sync.start();\n'
                     '\t{name}(').format(name=fun)
            for idx in range(len(kernels[fun])):
                code += '\n\t\t*reinterpret_cast<{type}*>(args[{idx}]),'.format(type=kernels[fun][idx], idx=idx)
            if len(kernels[fun])>0:
                code = code[:-1]
            code += ('\n\t);\n'
                     '\tblock_sync.end();\n'
                     '}}\n'
                     '\n').format(name=fun)
        return code

# ---------- end of builder class ----------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("no file given")
        exit()

    code = Builder().generate_ptx_code(sys.argv[1])

    baseFileName=os.path.basename(sys.argv[1])
    ccFileName=baseFileName+'.cc'

    with open(ccFileName,mode='w') as f:
        print(code,file=f)

    ptxFileName=None

    if baseFileName.endswith('.cu'):
        ptxFileName=baseFileName[:-3]+'.ptx'
    else:
        ptxFileName=baseFileName+'.ptx'

    with open(ccFileName) as f:
        subprocess.check_call(['g++','-std=c++11','-O3','-Wall','-march=native','-shared','-fPIC','-xc++','-pipe','-o'+ptxFileName,'-'],stdin=f)
