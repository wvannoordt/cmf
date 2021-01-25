#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
#define LEVEL 2
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::CreateParallelContext(&argc, &argv);
    
    size_t nguard = 3;
    size_t arSize = 5;
    int* dataArray = (int*)malloc((arSize+2*nguard)*sizeof(int));
    cmf::DataExchangePattern dataHandler(&cmf::globalGroup);
    
    int leftNeighbor = cmf::globalGroup.Rank()-1;
    if (leftNeighbor < 0) leftNeighbor += cmf::globalGroup.Size();
    int rightNeighbor = cmf::globalGroup.Rank()+1;
    if (rightNeighbor >= cmf::globalGroup.Size()) rightNeighbor -= cmf::globalGroup.Size();
    
    //left transaction
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray+nguard), nguard*sizeof(int), cmf::globalGroup.Rank(), leftNeighbor));
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray), nguard*sizeof(int), leftNeighbor, cmf::globalGroup.Rank()));
    
    //right transaction
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray+arSize), nguard*sizeof(int), cmf::globalGroup.Rank(), rightNeighbor));
    dataHandler.Add(new cmf::SingleTransaction((void*)(dataArray+arSize+nguard), nguard*sizeof(int), rightNeighbor, cmf::globalGroup.Rank()));
    
    dataHandler.ExchangeData();
    
    free(dataArray);
    return 0;
}