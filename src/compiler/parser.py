class Parser(object):
    def set_kernel_declaration_callback(self, kernel_declaration_callback):
        self.k = kernel_declaration_callback

    def set_shared_memory_definition_callback(self, shared_memory_definition_callback):
        self.m = shared_memory_definition_callback

    def parse(self, file):
        result = ('extern "C"\n'
                  '{\n'
                  '\n'
                  'int fun1(int* argIntStar, float argFloat)\n'
                  '{\n'
        )
        result += self.m('fun1', 'tab', 'int[123]')
        result += self.m('fun1', 'temp', 'int')
        result += self.m('fun1', 'ptr', 'int*')
        result += self.m('fun1', 'arr', 'int[2]')
        result += ('\n'
                   '    ptr=&tab[5];\n'
                   '    tab[5]=argIntStar[threadIdx.x];\n'
                   '    argIntStar[6]=*ptr;\n'
                   '    return 10;\n'
                   '}\n'
                   '\n'
                   'void fun2(char* argCharStar,int argInt)\n'
                   '{\n'
        )
        result += self.m('fun2', 'tab', 'char[12]')
        result += self.m('fun2', 'a', 'char')
        result += self.m('fun2', 'b', 'char')
        result += self.m('fun2', 'c', 'char*')
        result += self.m('fun2', 'q', 'int')
        result += self.m('fun2', 'w', 'int[2]')
        result += self.m('fun2', 'e', 'int*')

        result += ('\n'
                   '}\n'
                   '\n'
                   '}'
        )

        self.k('fun1', ('int*', 'float'))
        self.k('fun2', ('char*', 'int'))

        return result