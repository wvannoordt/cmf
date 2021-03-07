#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
#include <chrono>
#include <unistd.h>

std::string fillstr(int val, int bsize)
{
    std::string output = std::to_string(val);
    while (output.length() < bsize)
    {
        output += " ";
    }
    return output;
}
struct ExchangeArr
{
    int* data;
    int ni;
    int nj;
    int nguard;
    ExchangeArr(int ni_in, int nj_in, int nguard_in)
    {
        ni = ni_in;
        nj = nj_in;
        nguard = nguard_in;
        data = (int*)malloc((ni+2*nguard)*(nj+2*nguard)*sizeof(int));
    }
    void Fill(int val)
    {
        for (int i = 0; i < (ni+2*nguard)*(nj+2*nguard); i++)
        {
            data[i] = 0;
        }
    }
    void Print(void)
    {
        for (int j = -nguard; j < nj + nguard; j++)
        {
            std::string line = "";
            for (int i = -nguard; i < ni + nguard; i++)
            {
                line += fillstr((*this)(i,j), 7);
            }
            std::cout << line << std::endl;
        }
    }
    int& operator () (int i, int j)
    {
        return *(data + (i+nguard) + (j+nguard)*(ni+2*nguard));
    }
    ~ExchangeArr(void)
    {
        free(data);
    }
};

int ij2n(int i, int j)
{
    return ((i+10)%2) + 2*((j+10)%2);
}

int n2i(int n)
{
    switch (n)
    {
        case 0: return 0;
        case 1: return 1;
        case 2: return 0;
        case 3: return 1;
    }
}

int n2j(int n)
{
    switch (n)
    {
        case 0: return 0;
        case 1: return 0;
        case 2: return 1;
        case 3: return 1;
    }
}

void GetSendIndices(int neighType, ExchangeArr& arr, int* imin_out, int* imax_out, int* jmin_out, int* jmax_out)
{
    //internal to block
    switch (neighType)
    {
        case 0:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 1:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.ni;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 2:
        {
            *imin_out = arr.ni;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 3:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nj;
            return;
        }
        case 4:
        {
            *imin_out = arr.ni;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nj;
            return;
        }
        case 5:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nj;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 6:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.ni;
            *jmin_out = arr.nj;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 7:
        {
            *imin_out = arr.ni;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nj;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
    }
}

void GetRecvIndices(int neighType, ExchangeArr& arr, int* imin_out, int* imax_out, int* jmin_out, int* jmax_out)
{
    //external to block
    switch (neighType)
    {
        case 0:
        {
            *imin_out = 0;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = 0;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 1:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.ni;
            *jmin_out = 0;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 2:
        {
            *imin_out = arr.ni+arr.nguard;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = 0;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 3:
        {
            *imin_out = 0;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nj;
            return;
        }
        case 4:
        {
            *imin_out = arr.ni + arr.nguard;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nguard;
            *jmax_out = *jmin_out + arr.nj;
            return;
        }
        case 5:
        {
            *imin_out = 0;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nj + arr.nguard;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 6:
        {
            *imin_out = arr.nguard;
            *imax_out = *imin_out + arr.ni;
            *jmin_out = arr.nj + arr.nguard;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
        case 7:
        {
            *imin_out = arr.ni + arr.nguard;
            *imax_out = *imin_out + arr.nguard;
            *jmin_out = arr.nj + arr.nguard;
            *jmax_out = *jmin_out + arr.nguard;
            return;
        }
    }
}

void GetExchangeInfo(int neighType, ExchangeArr& arr, std::vector<size_t>* offsetsSend_out, std::vector<size_t>* sizesSend_out, std::vector<size_t>* offsetsRecv_out, std::vector<size_t>* sizesRecv_out)
{
    int iminR, imaxR, jminR, jmaxR;
    int iminS, imaxS, jminS, jmaxS;
    GetRecvIndices(neighType, arr, &iminR, &imaxR, &jminR, &jmaxR);
    GetSendIndices(7-neighType, arr, &iminS, &imaxS, &jminS, &jmaxS);
    int deltaIR = imaxR - iminR;
    int deltaIS = imaxS - iminS;
    if (deltaIS != deltaIR)
    {
        std::cout << "Dang, looks like deltaIR != deltaIS..." << std::endl;
        abort();
    }
    for (int j = jminR; j < jmaxR; j++)
    {
        offsetsRecv_out->push_back((iminR + j*(arr.ni+2*arr.nguard))*sizeof(int));
        sizesRecv_out->push_back(deltaIR*sizeof(int));
    }
    for (int j = jminS; j < jmaxS; j++)
    {
        offsetsSend_out->push_back((iminS + j*(arr.ni+2*arr.nguard))*sizeof(int));
        sizesSend_out->push_back(deltaIS*sizeof(int));
    }
}
int main(int argc, char** argv)
{
    EXIT_WARN_IF_PARALLEL;
    EXIT_WARN_IF_DIM_NOT(3);
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CreateParallelContext(&argc, &argv);
    int nguard = 3;
    int ni = 8;
    int nj = 8;
    
    ExchangeArr array(ni, nj, nguard);
    array.Fill(0);
    int count = 1;
    for (int j = 0; j < nj; j++)
    {
        for (int i = 0; i < ni; i++)
        {
            array(i, j) = 1000*(cmf::globalGroup.Rank()+1) + (count++);
        }
    }
    int my_i = n2i(cmf::globalGroup.Rank());
    int my_j = n2j(cmf::globalGroup.Rank());
    std::vector<int> neighs;
    neighs.push_back(ij2n(my_i-1, my_j-1));
    neighs.push_back(ij2n(my_i,   my_j-1));
    neighs.push_back(ij2n(my_i+1, my_j-1));
    neighs.push_back(ij2n(my_i-1, my_j));
    neighs.push_back(ij2n(my_i+1, my_j));
    neighs.push_back(ij2n(my_i-1, my_j+1));
    neighs.push_back(ij2n(my_i,   my_j+1));
    neighs.push_back(ij2n(my_i+1, my_j+1));
    if (cmf::globalGroup.IsRoot())
    {
        std::cout << "============= BEFORE EXCHANGE =============" << std::endl;
    }
    cmf::globalGroup.Synchronize();
    usleep(10000*cmf::globalGroup.Rank());
    array.Print();
    std::cout << std::endl;
    cmf::globalGroup.Synchronize();
    
    cmf::DataExchangePattern arrayExchange(&cmf::globalGroup);
    
    // 0 | 1 | 2
    // 3 |   | 4
    // 5 | 6 | 7
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 2; i++)
        {
            int currentRankOnBlock = ij2n(i, j);
            for (int n = 0; n < neighs.size(); n++)
            {
                int neighborRank = neighs[n];
                std::vector<size_t> offsetsSend;
                std::vector<size_t> sizesSend;
                std::vector<size_t> offsetsRecv;
                std::vector<size_t> sizesRecv;
                GetExchangeInfo(n,   array, &offsetsSend, &sizesSend, &offsetsRecv, &sizesRecv);
                
                //current block sends to neighbor
                arrayExchange.Add(new cmf::MultiTransaction((void*)array.data, offsetsSend, sizesSend, currentRankOnBlock, neighborRank));
                //neighbor block receives from current
                arrayExchange.Add(new cmf::MultiTransaction((void*)array.data, offsetsRecv, sizesRecv, neighborRank, currentRankOnBlock));
            }
        }
    }
    
    arrayExchange.ExchangeData();
    
    cmf::globalGroup.Synchronize();
    if (cmf::globalGroup.IsRoot())
    {
        std::cout << "============= AFTER  EXCHANGE =============" << std::endl;
    }
    cmf::globalGroup.Synchronize();
    usleep(10000*cmf::globalGroup.Rank());
    array.Print();
    std::cout << std::endl;
    return 0;
}