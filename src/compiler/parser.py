class Parser(object):
    def set_kernel_declaration_callback(self, kernel_declaration_callback):
        self.k = kernel_declaration_callback

    def set_shared_memory_definition_callback(self, shared_memory_definition_callback):
        self.m = shared_memory_definition_callback

    def parse(self, file):
        self.k('kern1', ('int*', 'int*', 'char'))
        self.k('kern2', ('int', 'char*'))
        self.m('kern1', 'a', 'int*')
        self.m('kern1', 'b', 'int[5]')
        self.m('kern1', 'c', 'char')
        self.m('kern2', 'd', 'int*')
        self.m('kern2', 'e', 'int[5]')
        self.m('kern2', 'f', 'char[1]')

        return ""