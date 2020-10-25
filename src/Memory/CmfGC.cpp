#include "CmfGC.h"
#include "CmfScreen.h"
namespace cmf
{
    CmfGC cmfGC;
    CmfGC::CmfGC(void){}
    
    CmfGC::~CmfGC(void)
    {
        WriteLineStd(4, "Freeing resources");
    }
}