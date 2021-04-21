#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
#define DATASIZE 2
class fhash : public cmf::ICmfHashable {};
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    fhash writeHash;
    fhash readHash;
    auto& glob = cmf::globalGroup;
    for (int mode = 0; mode < 2; mode++)
    {
        cmf::ParallelFile ofile(&glob);
        ofile.Open("output/testFile.par");
        for (int i = 0; i < glob.Size(); i++)
        {
            ofile.SetSerialRank(i);
            if (mode==0) //write
            {
                std::string message = strformat("Hello from {}", glob.Rank());
                writeHash.AugmentHash(message);
                ofile.SerialWrite(message);
            }
            else
            {
                std::string message = ofile.SerialRead();
                readHash.AugmentHash(message);
            }
        }
        
        if (mode==0) //write
        {
            std::string oneMessage = strformat("This is a message that all ranks should see: pi = {}", 3.1415);
            ofile.Write(oneMessage);
            writeHash.AugmentHash(oneMessage);
        }
        else
        {
            std::string oneMessage = ofile.Read();
            writeHash.AugmentHash(oneMessage);
        }

        int dataToWrite[DATASIZE];
        if (mode==0) //write
        {
            for (int i = 0; i < DATASIZE; i++)
            {
                dataToWrite[i] = glob.Rank();
                writeHash.AugmentHash(dataToWrite[i]);
            }
        }
        
        cmf::ParallelDataBuffer parDataBuf;
        parDataBuf.Add(&dataToWrite[0], DATASIZE, glob.Rank()*DATASIZE);
        
        if (mode==0) //write
        {
            ofile.ParallelWrite(parDataBuf);
        }
        else
        {
            ofile.ParallelRead(parDataBuf);
            for (int i = 0; i < DATASIZE; i++)
            {
                readHash.AugmentHash(dataToWrite[i]);
            }
        }
    }
    
    print("Rank/hash(R)/hash(W):", glob.Rank(), readHash.GetHash(), writeHash.GetHash());
    
    return 0;
}