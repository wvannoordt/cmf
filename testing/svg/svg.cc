#include "cmf.h"
#include "cmftestutils.h"

int main(int argc, char** argv)
{
    cmf::ReadInput("input.ptl");
    PTL::PropertySection user = cmf::mainInput["User"];
    int a;
    user["a"].MapTo(&a) = new PTL::Variables::PTLInteger(0, "A dummy variable");
    user.StrictParse();
    WriteLine(1, "a="+std::to_string(a));
    return 0;
}