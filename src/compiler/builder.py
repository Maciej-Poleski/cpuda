#!/usr/bin/env python3
import sys

from parser import Parser


class ObjectInfo(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type


kernels = {}
shared_objects = {}
code = ""


def kernel_handler(name, args):
    kernels[name] = args


def shared_memory_handler(fun_name, var_name, var_type):
    if fun_name not in shared_objects:
        shared_objects[fun_name] = []
    shared_objects[fun_name] += [ObjectInfo(var_name, var_type)]
    return 'auto& {var_name} = detail::shared::{fun_name}[detail::block_flat_idx()]->{var_name};\n'.format(
        fun_name=fun_name,
        var_name=var_name)


parser = Parser()
parser.set_kernel_declaration_callback(kernel_handler)
parser.set_shared_memory_definition_callback(shared_memory_handler)

with open('templates/header_constant.cc') as f:
    code += f.read()

result = parser.parse(sys.argv[1])

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
         '};\n')

code += result

code += ('\n#include <type_traits>\n'
         '\n'
         'extern "C" void _kernel_global_init(detail::gridDim_t gridDim)\n'
         '{')

for fun in shared_objects:
    code += ('\n'
             '\tusing {name}_t = std::remove_pointer<decltype(detail::shared::{name})>::type;\n'
             '\tdetail::shared::{name} = new {name}_t[detail::grid_flat_size(gridDim)];\n').format(name=fun)

code += '}\n\n'

code += ("extern \"C\" void _kernel_global_deinit()\n"
         "{\n")
for fun in shared_objects:
    code += '\tdelete [] detail::shared::{name};\n'.format(name=fun)

code += '}\n\n'

code += ('extern "C" void _kernel_block_init(detail::gridDim_t gridDim, detail::blockIdx_t blockIdx)\n'
         '{\n')
for fun in shared_objects:
    code += ('\tusing {name}_t = std::remove_pointer<decltype(detail::shared::{name})>::type;\n'
             '\tdetail::shared::{name}[detail::block_flat_idx(gridDim, blockIdx)] = new std::remove_pointer<{name}_t>::type;\n'
    ).format(name=fun)
code += '}\n\n'

code += ('extern "C" void _kernel_block_deinit(detail::gridDim_t gridDim, detail::blockIdx_t blockIdx)\n'
         '{\n')
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
             '\t{name}(').format(name=fun)
    for idx in range(len(kernels[fun])):
        code += '\n\t\t*reinterpret_cast<{type}*>(args[{idx}]),'.format(type=kernels[fun][idx], idx=idx)
    code = code[:-1]
    code += ('\n\t);\n'
             '}}\n'
             '\n').format(name=fun)

print(code)
