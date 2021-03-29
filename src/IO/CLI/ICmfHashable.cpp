#include "ICmfHashable.h"

namespace cmf
{
    ICmfHashable::ICmfHashable(void)
    {
        hash = PRIMEH;
    }
    
    ICmfHashable::~ICmfHashable(void)
    {
        
    }
    
    void ICmfHashable::AugmentHash(int value)
    {
        hash = (hash * PRIMEA) ^ (size_t)(value * PRIMEB);
    }
    
    void ICmfHashable::AugmentHash(std::string value)
    {
        for (auto& c:value) this->AugmentHash((int)c);
    }
    
    size_t ICmfHashable::GetHash(void)
    {
        return hash;
    }
}