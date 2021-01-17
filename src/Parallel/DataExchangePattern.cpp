#include "DataExchangePattern.h"
#include "CmfScreen.h"
namespace cmf
{
    DataExchangePattern::DataExchangePattern(ParallelGroup* group_in)
    {
        group = group_in;
    }
    
    DataExchangePattern::~DataExchangePattern(void)
    {
        
    }
    
    void DataExchangePattern::ExchangeData(void)
    {
        WriteLine(0, "WARNING: DataExchangePattern::ExchangeData not fully implemented");
    }
}