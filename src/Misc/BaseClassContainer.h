#ifndef CMF_BASE_CLASS_CONTAINER_H
#define CMF_BASE_CLASS_CONTAINER_H
#include <vector>
#include <type_traits>
namespace cmf
{
    ///@brief Provides an interface for a container of abstract classes,
    ///preventing the use of the external "new" operator
    template <class BaseType> class BaseClassContainer
    {
        public:
            
            ///@brief Empty Constructor
            ///@author WVN
            BaseClassContainer(void){}
            
            ///@brief Adds a new instance of the specified type, returns the new pointer
            ///@author WVN
            template <class NewObjType, typename... ts> NewObjType* Add(ts... params)
            {
                static_assert(std::is_base_of<BaseType, NewObjType>::value, "Template type does not inherit from base type!");
                NewObjType* newObj = new NewObjType(params...);
                items.push_back(newObj);
                OnAdd(newObj);
                return newObj;
            }
            
            ///@brief Defines an overrideable callback after Add(...) is called
            ///@param newItem The newly created item
            ///@author WVN
            virtual void OnAdd(BaseType* newItem) {}
            
            ///@brief Destrcutor that deletes all contained instances
            ///@author WVN
            ~BaseClassContainer(void)
            {
                for (auto& p:items)
                {
                    delete p;
                }
            }
            
            /// @brief Returns item of the specified type
            /// @author WVN
            template <class SearchType> std::vector<SearchType*> GetItemsByType(void)
            {
                std::vector<SearchType*> output;
                static_assert(std::is_base_of<BaseType, SearchType>::value, "Template type does not inherit from base type!");
                for (auto t:items)
                {
                    SearchType* ptr = dynamic_cast<SearchType*>(t);
                    if (ptr != NULL)
                    {
                        output.push_back(ptr);
                    }
                }
                return output;
            }
            
            /// @brief Indexing operator
            /// @param i index
            /// @author WVN
            BaseType* operator [] (int i) {return items[i];}
            
            /// @brief Begin() overload for range iteration
            /// @author WVN
            typename std::vector<BaseType*>::iterator begin() noexcept
            {
                return items.begin();
            }
            
            /// @brief const Begin() overload for range iteration
            /// @author WVN
            typename std::vector<BaseType*>::const_iterator begin() const noexcept
            {
                return items.begin();
            }
            
            /// @brief End() overload for range iteration
            /// @author WVN
            typename std::vector<BaseType*>::iterator end() noexcept
            {
                return items.end();
            }
            
            /// @brief constant End() overload for range iteration
            /// @author WVN
            typename std::vector<BaseType*>::const_iterator end() const noexcept
            {
                return items.end();
            }
            
        protected:
            std::vector<BaseType*> items;
    };
}

#endif