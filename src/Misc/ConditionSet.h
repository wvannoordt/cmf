#ifndef CMF_CONDITION_SET_H
#define CMF_CONDITION_SET_H
#include <map>
#include <string>
namespace cmf
{
    /// @brief A class representing set of logical conditions with string names
    /// @author WVN
    class ConditionSet
    {    
        public:
            /// @brief Empty constructor
            /// @author WVN
            ConditionSet(void);
            
            /// @brief Empty destructor
            /// @author WVN
            ~ConditionSet(void);
            
            /// @brief Adds a required condition
            /// @param name The name of the condition
            /// @author WVN
            void AddCondition(std::string name);
            
            /// @brief Checks if a condition exists
            /// @author WVN
            bool HasCondition(std::string name);
            
            /// @brief Returns the value of a condition
            /// @author WVN
            bool GetCondition(std::string name);
            
            bool CheckAll(void);
            
        private:
            
            /// @brief The list containing the conditions
            std::map<std::string, bool> conditions;
    };
}

#endif