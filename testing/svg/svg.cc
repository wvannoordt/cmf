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
    cmf::SvgLine* line = image.AddLine(100, 100, 900, 900);
    cmf::SvgRectangle* rect1 = image.AddRectangle(50, 74, 300, 300);
    cmf::SvgRectangle* rect2 = image.AddRectangle(100, 100, 900, 900);
    // cmf::SvgRectangle* rect3 = image["greens"].AddRectangle(588, 189, 839, 505);
    // cmf::SvgRectangle* rect4 = image["greens"].AddRectangle(270, 189, 450, 505);
    cmf::SvgRectangle* rect3 = image["layer1"].AddRectangle(588, 189, 839, 505);
    cmf::SvgRectangle* rect4 = image["layer1"].AddRectangle(270, 189, 450, 505);
    // image["layer1"].SetVisibility(false);
    
    rect3->SetFillColor("green");
    rect4->SetFillColor("green");
    rect1->BringToFront();
    // rect3->SendToBack();
    // rect4->SendToBack();
    image["layer1"].SendToBack();
    
    rect2->SetFillOpacity(0.4);
    rect2->SetFillColor("red");
    rect1->SetFillColor("blue");
    image.SetFillColor("white");
    image.Write("output/test.svg");
    return 0;
}