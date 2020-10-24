#ifndef CMF_TEST_UTILS_H
#define CMF_TEST_UTILS_H


#define EXIT_WARN_IF_DIM_NOT(myint) if (myint != cmf::GetDim())\
{\
    cmf::cmfout << "\n--------------------------------------------------------------------------------" << cmf::cmfendl;\
    cmf::cmfout << "WARNING: skipping test case in file " << __FILE__ << ": dimensions incompatible." << cmf::cmfendl;\
    cmf::cmfout << "--------------------------------------------------------------------------------\n" << cmf::cmfendl;\
    return 0;\
}\

#endif