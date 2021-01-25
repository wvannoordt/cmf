#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
#include <unistd.h>
#define LEVEL 2
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::CreateParallelContext(&argc, &argv);
    size_t nguard = 3;
    size_t arSize = 10;
    int* dataArray = (int*)malloc((arSize+2*nguard)*sizeof(int));
    cmf::DataExchangePattern dataHandler(&cmf::globalGroup);
    
    for (int i = 0; i < arSize + 2*nguard; i++) dataArray[i] = -1;
    for (int i = nguard; i < arSize + nguard; i++) dataArray[i] = 10+10*cmf::globalGroup.Rank() - nguard + i;
    
    int leftNeighbor = cmf::globalGroup.Rank() - 1;
    if (leftNeighbor < 0) leftNeighbor += cmf::globalGroup.Size();
    int rightNeighbor = cmf::globalGroup.Rank() + 1;
    if (rightNeighbor >= cmf::globalGroup.Size()) rightNeighbor -= cmf::globalGroup.Size();
    
    for (int p = 0; p < cmf::globalGroup.Size(); p++)
    {
        if (p == cmf::globalGroup.Rank())
        {
            std::cout << "Rank " << p << ", before:\n";
            for (int i = 0; i < arSize+2*nguard; i++)
            {
                std::cout << dataArray[i] << " ";
            }
            std::cout << "\n";
        }
        usleep(1000);
        cmf::globalGroup.Synchronize();
    }
    cmf::globalGroup.Synchronize();
    if (cmf::globalGroup.IsRoot()) std::cout << std::endl;
    // ctor: target, size, sender, receiver
    
    //left transaction
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray+nguard), nguard*sizeof(int), cmf::globalGroup.Rank(), leftNeighbor));
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray), nguard*sizeof(int), leftNeighbor, cmf::globalGroup.Rank()));
    
    //right transaction
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray+arSize), nguard*sizeof(int), cmf::globalGroup.Rank(), rightNeighbor));
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray+arSize+nguard), nguard*sizeof(int), rightNeighbor, cmf::globalGroup.Rank()));
    
    dataHandler.ExchangeData();
    
    for (int p = 0; p < cmf::globalGroup.Size(); p++)
    {
        if (p == cmf::globalGroup.Rank())
        {
            std::cout << "Rank " << p << ", after:\n";
            for (int i = 0; i < arSize+2*nguard; i++)
            {
                std::cout << dataArray[i] << " ";
            }
            std::cout << "\n";
            usleep(1000);
        }
        cmf::globalGroup.Synchronize();
    }
    
    free(dataArray);
    return 0;
}