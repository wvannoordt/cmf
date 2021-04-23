#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
using cmf::print;
using cmf::strformat;
#define DATASIZE 100
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
    cmf::ParallelFile wfile(&glob);
    wfile.Open("output/testFile.par");
    std::string message1w = strformat("The value of pi is {}", 3.14159);
    std::string message2w = strformat("This is just some other string, not sure what to write here {}", "...");
    wfile.Write(message1w);
    wfile.Write(message2w);
    writeHash.AugmentHash(message1w);
    writeHash.AugmentHash(message2w);
    
    for (int i = 0; i < glob.Size(); i++)
    {
        wfile.SetSerialRank(i);
        std::string mymessage = strformat("my processor rank is {}", glob.Rank());
        wfile.SerialWrite(mymessage);
        if (i==glob.Rank()) writeHash.AugmentHash(mymessage);
    }
    
    int wdata[DATASIZE];
    for (int i = 0; i < DATASIZE; i++)
    {
        wdata[i] = i + 9*glob.Rank();
        writeHash.AugmentHash(wdata[i]);
    }
    cmf::ParallelDataBuffer dataBuf;
    dataBuf.Add(&wdata[0], DATASIZE, DATASIZE*glob.Rank());
    wfile.ParallelWrite(dataBuf);
    
    std::string finalMessage = "this is the final message in the file";
    wfile.Write(finalMessage);
    writeHash.AugmentHash(finalMessage);
    
    wfile.Close();
    
    
    cmf::ParallelFile rfile(&glob);
    rfile.Open("output/testFile.par");
    std::string message1r = rfile.Read();
    std::string message2r = rfile.Read();
    readHash.AugmentHash(message1r);
    readHash.AugmentHash(message2r);
    
    for (int i = 0; i < glob.Size(); i++)
    {
        rfile.SetSerialRank(i);
        std::string mymessage = rfile.SerialRead();
        if (i==glob.Rank()) readHash.AugmentHash(mymessage);
    }
    
    int rdata[DATASIZE];
    cmf::ParallelDataBuffer dataBufr;
    dataBufr.Add(&rdata[0], DATASIZE, DATASIZE*glob.Rank());
    rfile.ParallelRead(dataBufr);
    for (int i = 0; i < DATASIZE; i++)
    {
        readHash.AugmentHash(rdata[i]);
    }
    
    std::string finalMessage2 = rfile.Read();
    readHash.AugmentHash(finalMessage2);
    
    print(readHash.GetHash(), writeHash.GetHash());
    size_t hdiff = (readHash.GetHash()-writeHash.GetHash());
    
    glob.Synchronize();
    
    if (!glob.HasSameValue(hdiff))
    {
        if (glob.IsRoot())
        {
            print("Parallel IO hashes do not match on at least one processor... not a good thing");
        }
    }
    
    rfile.Close();
    
    return (readHash.GetHash()==writeHash.GetHash())?0:1;
}