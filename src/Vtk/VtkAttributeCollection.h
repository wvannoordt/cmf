#ifndef VTK_ATTR_COLLEC_H
#define VTK_ATTR_COLLEC_H
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>
#include "VtkAttributable.h"

namespace cmf
{
    /// @brief Class representing a collection of VtkAttributable objects
    /// @author WVN
    class VtkAttributeCollection
    {
        public:
            /// @brief Constructor for VtkAttributeCollection
            /// @author WVN
            VtkAttributeCollection(void){cleared = false; count=0;}

            /// @brief Throws error of any required attribute is not found.
            /// @author WVN
            void EnforceAllRequiredAttributes(void)
            {
                for (std::map<std::string, VtkAttributable*>::iterator it = attributables.begin(); it!=attributables.end(); it++)
                {
                    it->second->EnforceAllRequiredAttributes();
                }
            }

            /// @brief Check to see if an attribute exists in the provided attributes
            /// @param name Name of the attribute to check
            /// @author WVN
            bool Exists(std::string name)
            {
                std::map<std::string,VtkAttributable*>::iterator it = attributables.find(name);
                if(it != attributables.end())
                {
                   return true;
                }
                return false;
            }
            
            /// @brief Add an attribute to the current VtkAttributable
            /// @param name Name of the attribute to add
            /// @param newType The type of the attribute data
            /// \see VtkAttributableType
            /// @author WVN
            VtkAttributable* AddAttributable(std::string name, VtkAttributableType::VtkAttributableType newType)
            {
                if (Exists(name)) CmfError("Multiple definition of attribute " + name);
                attributables.insert({name, new VtkAttributable(name, newType)});
                indices.insert({count, name});
                count++;
                return attributables[name];
            }

            /// @brief Returns VtkAttributable with the given name.
            /// \pre Error is thrown if name does not exist.
            /// @param name Name of the attribute to retrieve
            /// @author WVN
            VtkAttributable* GetAttributable(std::string name)
            {
                if (!Exists(name)) CmfError("Attempted to fetch non-existent attribute " + name);
                return attributables[name];
            }

            /// @brief Destructor for the VtkAttributeCollection class, explicitly destroys all contained VtkAttributable objects.
            /// @author WVN
            ~VtkAttributeCollection(void)
            {
                if (!cleared)
                {
                    cleared = true;
                    for (std::map<std::string, VtkAttributable*>::iterator it = attributables.begin(); it!=attributables.end(); it++)
                    {
                        if (it->second != NULL)
                        {
                            it->second->Destroy();
                            delete it->second;
                            it->second = NULL;
                        }
                    }
                    attributables.clear();
                }
            }
            
            /// @brief Writes all of the contained VtkAttributable objects to the provided file stream
            /// @param myfile File stream to be written to
            /// @author WVN
            void WriteAll(std::ofstream & myfile)
            {
                for (int i = 0; i < count; i++)
                {
                    attributables[indices[i]]->Write(myfile);
                }
            }
        private:
            /// @brief List of VtkAttributable objects by name
            std::map<std::string, VtkAttributable*> attributables;
            
            /// @brief List of name by integer index
            std::map<int, std::string> indices;
            
            /// @brief Total number of VtkAttributable objects
            int count;
            
            /// @brief Set to true when destructor called
            bool cleared;
    };

    /// @brief A class to be inherited that provides a VtkAttributeCollection object
    /// \see VtkAttributeCollection VtkAttributable
    /// @author WVN
    class VtkHasAttributableCollection
    {
        public:
            /// @brief An empty (abstract) constructor
            /// @author WVN
            VtkHasAttributableCollection(void){}
            
            /// @brief An empty (abstract) destructor
            /// @author WVN
            ~VtkHasAttributableCollection(void){}
            
            /// @brief Returns a VtkAttributable by name
            /// @param name The name of the VtkAttributable to return
            /// @author WVN
            VtkAttributable* Component(std::string name)
            {
                return collection.GetAttributable(name);
            }
            
            /// @brief Ensures that the underlying VtkAttributeCollection has all required attributes
            /// \pre Throws an error if not all required attributes are found.
            /// @author WVN
            void CheckHasAllAttributes(void)
            {
                collection.EnforceAllRequiredAttributes();
            }
        protected:
            
            /// @brief The underlying VtkAttributeCollection
            VtkAttributeCollection collection;
    };
}

#endif
