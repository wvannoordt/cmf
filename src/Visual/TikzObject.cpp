#include <string>
#include "TikzObject.h"
#include <iostream>
#include <fstream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

namespace gTree
{
    TikzObject::TikzObject(void)
    {
        imageScale = 1.0;
    }

    TikzObject::~TikzObject(void)
    {

    }

    void TikzObject::Open(std::string filename)
    {
        myfile.open(filename.c_str());
        myfile << "\\documentclass[tikz,border=10pt]{standalone}" << std::endl;
        myfile << "\\usepackage{tikz}" << std::endl;
        myfile << "\\begin{document}" << std::endl;
        myfile << "\\begin{tikzpicture}" << std::endl;
    }

    void TikzObject::Close(void)
    {
        myfile << "\\end{tikzpicture}" << std::endl;
        myfile << "\\end{document}" << std::endl;
        myfile.close();
    }

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2)
    {DrawLine(x1, y1, x2, y2, "black", 1.0, "solid");}

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor)
    {DrawLine(x1, y1, x2, y2, TikzColorStr(lineColor), 1.0, "solid");}

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor, double lineThickness)
    {DrawLine(x1, y1, x2, y2, lineColor, lineThickness, "solid");}

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2, std::string lineColor, double lineThickness, TikzLineType::TikzLineType lineType)
    {DrawLine(x1, y1, x2, y2, lineColor, lineThickness, TikzTypeStr(lineType));}

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor, double lineThickness, std::string lineType)
    {DrawLine(x1, y1, x2, y2, TikzColorStr(lineColor), lineThickness, lineType);}

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor, double lineThickness, TikzLineType::TikzLineType lineType)
    {DrawLine(x1, y1, x2, y2, TikzColorStr(lineColor), lineThickness, TikzTypeStr(lineType));}

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2, std::string lineColor, double lineThickness, std::string lineType)
    {
        myfile << "\\draw[" << lineColor << ", " << lineType << ", line width=" << std::to_string(lineThickness) << "] ";
        myfile << "(" << std::to_string(imageScale*x1) << "," << std::to_string(imageScale*y1) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(imageScale*x2) << "," << std::to_string(imageScale*y2) << ");";
        myfile << std::endl;
    }
}
