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
    class VtkAttributeCollection
    {
        public:
            VtkAttributeCollection(void){cleared = false; count=0;}

            void EnforceAllRequiredAttributes(void)
            {
                for (std::map<std::string, VtkAttributable*>::iterator it = attributables.begin(); it!=attributables.end(); it++)
                {
                    it->second->EnforceAllRequiredAttributes();
                }
            }

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
                if (Exists(name)) CmfError("Multiple definition of attribute " + name);
                attributables.insert({name, new VtkAttributable(name, newType)});
                indices.insert({count, name});
                count++;
                return attributables[name];
            }

            VtkAttributable* GetAttributable(std::string name)
            {
                if (!Exists(name)) CmfError("Attempted to fetch non-existent attribute " + name);
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

            void WriteAll(std::ofstream & myfile)
            {
                for (int i = 0; i < count; i++)
                {
                    attributables[indices[i]]->Write(myfile);
                }
            }
        private:
            std::map<std::string, VtkAttributable*> attributables;
            std::map<int, std::string> indices;
            int count;
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
            void CheckHasAllAttributes(void)
            {
                collection.EnforceAllRequiredAttributes();
            }
        protected:
            VtkAttributeCollection collection;
    };
}

#endif
