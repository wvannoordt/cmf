#include "cmf.h"
#include "cmftestutils.h"

int main(int argc, char** argv)
{
    cmf::ReadInput("input.ptl");
    PTL::PropertySection user = cmf::mainInput["User"];
    double* imageBounds;
    user["imageBounds"].MapTo(&imageBounds) = new PTL::Variables::PTLStaticDoubleArray(4, "The bounds of the image");
    user.StrictParse();
    
    cmf::SvgImage image(imageBounds[0], imageBounds[1], imageBounds[2], imageBounds[3]);
    image.AddLine(100, 100, 900, 900);
    cmf::SvgRectangle* rect2 = image.AddRectangle(50, 74, 300, 300);
    cmf::SvgRectangle* rect = image.AddRectangle(100, 100, 900, 900);
    rect->SetFillOpacity(0.4);
    rect->SetFillColor("red");
    rect2->SetFillColor("green");
    image.SetFillColor("white");
    image.Write("output/test.svg");
    return 0;
}