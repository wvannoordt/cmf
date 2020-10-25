#include "ConditionSet.h"
#include <iostream>
namespace cmf
{
    ConditionSet::ConditionSet(void){}
    
    ConditionSet::~ConditionSet(void){}
    
    void ConditionSet::AddCondition(std::string name)
    {
        conditions.insert({name, false});
    }
    
    bool ConditionSet::HasCondition(std::string name)
    {
        return (conditions.find(name)!=conditions.end());
    }
        
    bool ConditionSet::GetCondition(std::string name)
    {
        if (!HasCondition(name)) return false;
        return conditions[name];
    }
    
    void ConditionSet::SetCondition(std::string name, bool conditionValue)
    {
        if (HasCondition(name))
        {
            conditions[name] = conditionValue;
        }
    }
    
    void ConditionSet::WriteUnmetConditions(std::string& output)
    {
        for (std::map<std::string, bool>::iterator it = conditions.begin(); it != conditions.end(); it++)
        {
            if (!it->second)
            {
                output = output + (it->first+"\n");
            }
        }
    }
        
    bool ConditionSet::CheckAll(void)
    {
        bool output = true;
        for (std::map<std::string, bool>::iterator it = conditions.begin(); it != conditions.end(); it++)
        {
            output = output && (it->second); 
        }
        return output;
    }
}