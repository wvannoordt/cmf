#ifndef CMF_OBJECT_LIST_H
#define CMF_OBJECT_LIST_H
#include <map>
#include <vector>
#include <type_traits>
namespace cmf
{
    ///@brief A class that acts as a hybrid between a vector and a map. Used when you need to iterate over objects in 
    ///the order they were added in, but also need to find them in logarithmic time. This class is not performant.
    ///@author WVN
    template <class listType> class ObjectList
    {
        public:
            ///@brief Empty constructor
            ///@author WVN
            ObjectList(void)
            {
                static_assert(std::is_copy_constructible<listType>::value, "Required copy-constructible type \"listType\" for ObjectList<listType>");
            }
            
            ///@brief Empty destructor
            ///@author WVN
            ~ObjectList(void){}
            
            ///@brief Adds an object to the list
            ///@param newObject The object to add
            ///@author WVN
            size_t Add(listType newObject)
            {
                objVector.push_back(newObject);
                size_t end = objVector.size()-1;
                objIndexList.insert({newObject, end});
                return end;
            }
            
            /// @brief Begin() overload for range iteration
            /// @author WVN
            typename std::vector<listType>::iterator begin() noexcept
            {
                return objVector.begin();
            }
            
            /// @brief const Begin() overload for range iteration
            /// @author WVN
            typename std::vector<listType>::const_iterator begin() const noexcept
            {
                return objVector.begin();
            }
            
            /// @brief End() overload for range iteration
            /// @author WVN
            typename std::vector<listType>::iterator end() noexcept
            {
                return objVector.end();
            }
            
            /// @brief constant End() overload for range iteration
            /// @author WVN
            typename std::vector<listType>::const_iterator end() const noexcept
            {
                return objVector.end();
            }
            
            ///@brief Checks to see if the list contains the object
            ///@param obj the object to check for
            ///@author WVN
            bool Has(listType obj)
            {
                return (objIndexList.find(obj) != objIndexList.end());
            }
            
            ///@brief Returns the size of the list
            ///@author WVN
            size_t Size(void)
            {
                return objVector.size();
            }
            
            ///@brief Indexing operator
            ///@param idx Index for the list
            ///@author WVN
            listType& operator [] (size_t idx)
            {
                return objVector[idx];
            }
            
            ///@brief Indexing operator, returns the index of the object in the list, or -1 if the list doesn't
            ///contain the object
            ///@param idx Index for the list
            ///@author WVN
            size_t& operator [] (listType idx)
            {
                if (!this->Has(idx)) return -1;
                return objIndexList[idx];
            }
            
        private:
            
            ///@brief The vector of objects
            std::vector<listType> objVector;
            
            ///@brief The map from objects to their indices in the list
            std::map<listType, size_t> objIndexList;
    };
};

#endif