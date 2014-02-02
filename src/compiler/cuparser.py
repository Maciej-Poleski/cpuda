import re

class Parser(object):
    def set_kernel_declaration_callback(self, kernel_declaration_callback):
        self.k = kernel_declaration_callback

    def set_shared_memory_definition_callback(self, shared_memory_definition_callback):
        self.m = shared_memory_definition_callback

    def parse(self, file):
        result = ('#define __device__\n'
                  '#define __global__\n'
                  '#define __shared__\n')
        deviceFunctionRe=re.compile(r'__device__\s+.*\s+(.*)\(.*\)',re.IGNORECASE)
        globalFunctionRe=re.compile(r'__global__\s+void\s+(.*)\(((.*,)*(.*))\)',re.IGNORECASE)
        sharedRe=re.compile(r'^(\s*)__shared__\s+(.*);',re.IGNORECASE|re.MULTILINE)
        singleDeclarationRe=re.compile(r'((.*)\W)(\w+)',re.IGNORECASE)
        sharedMemoryDeclarationRe=re.compile(r'^((unsigned\s+|signed\s+|)(([^*\s])+))((.+,)*.*)$',re.IGNORECASE)
        typedNamePartRe=re.compile(r'^(\W*)(\w*)(.*)$',re.IGNORECASE)
        function=None
        with open(file) as f:
            seen=''
            oldEnd=0
            for line in f:
                seen+=line
                match=None
                for match in deviceFunctionRe.finditer(seen):
                    pass
                if match and match.end()>oldEnd:
                    oldEnd=match.end()
                    function=match.group(1)
                    result+=line
                    continue
                match=None
                for match in globalFunctionRe.finditer(seen):
                    pass
                if match and match.end()>oldEnd:
                    oldEnd=match.end()
                    function=match.group(1)
                    args=match.group(2).split(',')
                    parsedArgs=[]
                    for arg in args:
                        am=singleDeclarationRe.match(arg)
                        if am:
                            parsedArgs+=[am.group(1).strip()]
                    result+=line
                    self.k(function,parsedArgs)
                    continue
                match=None
                for match in sharedRe.finditer(seen):
                    pass
                if match and match.end()>oldEnd:
                    oldEnd=match.end()
                    sm=sharedMemoryDeclarationRe.match(match.group(2))
                    typePrefix=sm.group(1)
                    args=sm.group(5).split(',')
                    for arg in args:
                        vm=typedNamePartRe.match(arg.strip())
                        result+=match.group(1)+self.m(function,vm.group(2),typePrefix+vm.group(1).strip()+vm.group(3).strip())
                    continue
                result+=line

        result += ('\n#undef __device__\n'
                   '#undef __global__\n'
                   '#undef __shared__\n')
        return result