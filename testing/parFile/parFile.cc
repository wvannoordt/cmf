#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
#define DATASIZE 2
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    
    auto& glob = cmf::globalGroup;
    cmf::ParallelFile ofile(&glob);
    ofile.Open("output/testFile.par");
    
    for (int i = 0; i < glob.Size(); i++)
    {
        ofile.SetSerialRank(i);
        ofile.SerialWrite("Hello from", glob.Rank());
    }

    int dataToWrite[DATASIZE];
    for (int i = 0; i < DATASIZE; i++)
    {
        dataToWrite[i] = glob.Rank();
    }
    
    cmf::ParallelDataBuffer parDataBuf;
    parDataBuf.Add(&dataToWrite[0], DATASIZE, glob.Rank()*DATASIZE);
    
    ofile.ParallelWrite(parDataBuf);
    
    ofile.SetSerialRank(1);
    ofile.SerialWrite("Hello again from", glob.Rank());
    
    return 0;
}