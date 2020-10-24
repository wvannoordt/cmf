#include "cmf.h"
#include "cmftestutils.h"
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::pxtype gray = {45, 45, 45, 255};
    cmf::pxtype t_red = {0, 0, 255, 145};
    cmf::pxtype white = {255, 255, 255, 255};
    cmf::PngImage img(600, 600);
    cmf::PngDrawer canvas(&img);
    canvas.SetCoordinateSystem(0.0, 1.0, 0.0, 1.0);
    canvas.Fill(gray);
    canvas.OutlineBox(0.7, 0.8, 0.8, 0.9, white, 0, 0);
    canvas.OutlineBox(0.2, 0.3, 0.8, 0.9, white, 0, 0);
    canvas.OutlineBox(0.8, 0.9, 0.3, 0.4, white, 0, 0);
    canvas.OutlineBox(0.1, 0.2, 0.3, 0.4, white, 0, 0);
    canvas.OutlineBox(0.1, 0.9, 0.2, 0.3, white, 0, 0);
    canvas.OutlineBox(0.3, 0.5, 0.05, 0.24, t_red, 0, 0);
    img.Write("output/test.png");
    return 0;
}