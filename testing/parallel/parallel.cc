#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    
    for (int p = 0; p < cmf::globalGroup.Size(); p++)
    {
        if (cmf::globalGroup.Rank() == p)
        {
            cmf::cmfout << "Output from process ID " << cmf::globalGroup.Rank() << ":" << cmf::cmfendl;
            cmf::mainInput.DebugPrint();
        }
        cmf::globalGroup.Synchronize();
    }
    return 0;
}