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

namespace Anaptric
{
    class VtkAttributeCollection
    {
        public:
            VtkAttributeCollection(void){cleared = false;}
            bool Exists(std::string name)
            {
                std::map<std::string,VtkAttributable*>::iterator it = attributables.find(name);
                if(it != attributables.end())
                {
                   return true;
                }
                return false;
            }
            VtkAttributable* AddAttributable(std::string name, VtkAttributableType::VtkAttributableType newType)
            {
                if (Exists(name)) __VTKERROR("Multiple definition of attribute " << name);
                attributables.insert({name, new VtkAttributable(name, newType)});
                return attributables[name];
            }

            VtkAttributable* GetAttributable(std::string name)
            {
                if (!Exists(name)) __VTKERROR("Attempted to fetch non-existent attribute " << name);
                return attributables[name];
            }

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
        private:
            std::map<std::string, VtkAttributable*> attributables;
            bool cleared;
    };

    class VtkHasAttributableCollection
    {
        public:
            VtkHasAttributableCollection(void){}
            ~VtkHasAttributableCollection(void){}
            VtkAttributable* Component(std::string name)
            {
                return collection.GetAttributable(name);
            }
        protected:
            VtkAttributeCollection collection;
    };
}

#endif
