#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    
    auto& glob = cmf::globalGroup;
    cmf::ParallelFile ofile(&glob);
    ofile.Open("output/testFile.par");
    
    for (int i = 0; i < 4; i++)
    {
        ofile.SetSerialRank(0);
        ofile.SerialWrite(strformat("Hello from {}", glob.Rank()));
    }
    
    return 0;
}