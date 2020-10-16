#ifndef CMF_ICMFGEOMETRY_H
#define CMF_ICMFGEOMETRY_H

namespace cmf
{
    class ICmfGeometry
    {
        ICmfGeometry(void) {geometryType = "NONE";}
        private:
            std::string geometryType;
    };
}

#endif