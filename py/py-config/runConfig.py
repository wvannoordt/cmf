import sys
import os
#CMF_DIM
#CMF_ENABLE_STACK_BLOB
#CMF_PARALLEL
#CMF_ZLIB_EXT_ENABLE
#CUDA_ENABLE
class UserInput:
    def __init__(self, args):
        self.args = args
        self.useExampleString = "./config"
        self.doHelp = ("-help" in args) or ("-h" in args)
        self.is3d = self.GetArg(name="3d", default=True, description="Compiles for 3D problems if true, 2D otherwise")
        self.isParallel = self.GetArg(name="mpi", default=True, description="Enables parallel computation with MPI, if installed")
        self.enableStackAllocation = self.GetArg(name="enableSimStack", default=False, description="Enables partial stack-allocation for large arrays (use \"false\" if unsure)")
        self.enableZlib = self.GetArg(name="zlib", default=True, description="Enables compilation with ZLIB (use \"true\" if unsure)")
        self.enableCuda = self.GetArg(name="cuda", default=True, description="Enables compilation with CUDA (use \"false\" if unsure)")
        self.optimizationLevel = self.GetArg(name="optimize", default=3, description="Optimization level for compilation")
        if (self.doHelp):
            print("------------------------------------------------------------")
            print("Example: " + self.useExampleString)
            sys.exit(0)
        
    def GetArg(self, **kwargs):
        if ("name" not in kwargs):
            print("Did not find required field \"name\".")
            sys.exit(1)
        if ("default" not in kwargs):
            print("Did not find required field \"default\".")
            sys.exit(1)
        if ("description" not in kwargs):
            print("Did not find required field \"description\".")
            sys.exit(1)
        if (self.doHelp):
            print("{}: {}".format(kwargs["name"].ljust(25), kwargs["description"]))
            self.useExampleString = self.useExampleString + " -" + kwargs["name"] + "=" + str(kwargs["default"]).lower()
        val = self.GetVal(kwargs["name"])
        if (val is None):
            return kwargs["default"]
        else:
            if (type(kwargs["default"])==type(True)):
                return val.lower() in ['true', '1', 't', 'y', 'yes']
            elif (type(kwargs["default"])==type(1)):
                return int(val)
            else:
                print("error: unsupported type of config arguement")
                sys.exit(1)
            return None
            
    def GetVal(self, name):
        for strng in self.args:
            if strng[0] == '-':
                strngs = strng[1:].split('=')
                nm = strngs[0]
                vl = strngs[1] if len(strngs)>1 else None
                if (nm==name):
                    return vl
        return None

class CmfConfig:
    def __init__(self, input):
        self.input = input
        self.configDict = {}
        if input.isParallel:
            CheckForMpi()
        if input.enableZlib:
            CheckForZlib()
        if input.enableCuda:
            CheckForCuda()
        self.GenerateConfigDict()
    
    def GenerateConfigDict(self):
        self.configDict["DIM"]                     = "3" if self.input.is3d else "2"
        self.configDict["PARALLEL"]                = "1" if self.input.isParallel else "0"
        self.configDict["ENABLE_STACK_ALLOCATION"] = "1" if self.input.enableStackAllocation else "0"
        self.configDict["ZLIB_ENABLE"]             = "1" if self.input.enableZlib else "0"
        self.configDict["CUDA_ENABLE"]             = "1" if self.input.enableCuda else "0"
        self.configDict["OPTLEVEL"]                = str(self.input.optimizationLevel)
        
        
    def WriteConfigFile(self, filename):
        ff = open(filename, 'w')
        for kv in self.configDict:
            ff.write("ifndef {}\n".format(kv))
            ff.write("export {}={}\n".format(kv, self.configDict[kv]))
            ff.write("endif\n\n")
        ff.close()
        
def CheckForMpi():
    print("Checking for MPI...")
    print("[TODO: write this function]")
    
def CheckForZlib():
    print("Check for Zlib...")
    print("[TODO: write this function]")

def CheckForCuda():
    print("Check for Cuda...")
    print("[TODO: write this function]")


def runConfig(args):
    input = UserInput(args)
    cmfConfig = CmfConfig(input)
    cmfConfig.WriteConfigFile('CmfBuildConfig.mk')
    

if (__name__ == "__main__"):
    runConfig(sys.argv[1:])