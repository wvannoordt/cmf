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
    
    size_t arSize = 100*sizeof(char);
    char* dataArray = (char*)malloc(arSize);
    cmf::DataExchangePattern dataBuf(&cmf::globalGroup);
    
    int leftNeighbor = cmf::globalGroup.Rank()-1;
    if (leftNeighbor < 0) leftNeighbor += cmf::globalGroup.Size();
    int rightNeighbor = cmf::globalGroup.Rank()+1;
    if (rightNeighbor >= cmf::globalGroup.Size()) rightNeighbor -= cmf::globalGroup.Size();
    
    dataBuf.Add(new cmf::SingleTransaction((void**)&dataArray, arSize, cmf::globalGroup.Rank(), leftNeighbor));
    free(dataArray);
    return 0;
}