#include "cmf.h"
#include "cmftestutils.h"

int main(int argc, char** argv)
{
    cmf::ReadInput("input.ptl");
    PTL::PropertySection user = cmf::mainInput["User"];
    double* imageBounds;
    user["imageBounds"].MapTo(&imageBounds) = new PTL::PTLStaticDoubleArray(4, "The bounds of the image");
    user.StrictParse();
    
    cmf::SvgImage image(imageBounds[0], imageBounds[1], imageBounds[2], imageBounds[3]);
    cmf::SvgNode lineStart(100.0,100.0);
    cmf::SvgNode lineEnd  (900.0,900.0);
    cmf::SvgLine* line = image["Items"].AddLine(&lineStart, &lineEnd);
    cmf::SvgNode ll1(50.0,74.0);
    cmf::SvgNode ur1(300.0,300.0);
    cmf::SvgRectangle* rect1 = image["Items"].AddRectangle(&ll1, &ur1);
    cmf::SvgRectangle* rect2 = image["Items"].AddRectangle(&lineStart, &lineEnd);
    // cmf::SvgRectangle* rect3 = image["greens"].AddRectangle(588, 189, 839, 505);
    // cmf::SvgRectangle* rect4 = image["greens"].AddRectangle(270, 189, 450, 505);
    cmf::SvgNode ll2(588.0,189.0);
    cmf::SvgNode ur2(839.0,505.0);
    cmf::SvgRectangle* rect3 = image["Items"].AddRectangle(&ll2, &ur2);
    
    cmf::SvgNode ll3(270.0,189.0);
    cmf::SvgNode ur3(450.0,505.0);
    cmf::SvgRectangle* rect4 = image["Items"].AddRectangle(&ll3, &ur3);
    
    cmf::SvgNode cirCenter(344.0, 200.6);
    double circleRadius = 50.0;
    cmf::SvgCircle* circle = image["Circles"].AddCircle(&cirCenter, circleRadius);
    // image["layer1"].SetVisibility(false);
    image["Circles"].SendToBack();
    
    rect3->SetFillColor("green");
    rect4->SetFillColor("green");
    // rect1->BringToFront();
    // rect3->SendToBack();
    // rect4->SendToBack();
    // image["layer1"].SendToBack();
    // image["Items"].SetVisibility(false);
    
    double low = 150;
    double high = 850;
    
    cmf::SvgNode p00(low,low);
    cmf::SvgNode p01(low,high);
    cmf::SvgNode p10(high,low);
    cmf::SvgNode p11(high,high);
    
    image["SillySquare"].AddLine(&p00, &p01);
    image["SillySquare"].AddLine(&p01, &p11);
    image["SillySquare"].AddLine(&p11, &p10);
    image["SillySquare"].AddLine(&p10, &p00);
    
    p00.x += 500.0;
    p00.y += 500.0;
    
    // image["SillySquare"].SetVisibility(false);
    
    rect2->SetFillOpacity(0.4);
    rect2->SetFillColor("red");
    rect1->SetFillColor("blue");
    image.SetFillColor("white");
    image.Write("output/test.svg");
    return 0;
}