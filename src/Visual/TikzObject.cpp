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
        PushLineType(TikzLineType::solid, 1.0, TikzColor::black);
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

    void TikzObject::PushLineType(TikzLineType::TikzLineType style, double thickness, TikzColor::TikzColor color)
    {PushLineType(TikzTypeStr(style), thickness, TikzColorStr(color));}

    void TikzObject::PushLineType(TikzLineType::TikzLineType style, double thickness, std::string color)
    {PushLineType(TikzTypeStr(style), thickness, color);}

    void TikzObject::PushLineType(std::string style, double thickness, TikzColor::TikzColor color)
    {PushLineType(style, thickness, TikzColorStr(color));}

    void TikzObject::PushLineType(std::string style, double thickness, std::string color)
    {
        lineStyleStack.push(style);
        lineColorStack.push(color);
        lineThicknessStack.push(thickness);
    }

    void TikzObject::PopLineType(void)
    {
        if (lineStyleStack.size()>1)
        {
            lineStyleStack.pop();
            lineColorStack.pop();
            lineThicknessStack.pop();
        }
    }

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2)
    {
        myfile << "\\draw[" << lineColorStack.top() << ", " << lineStyleStack.top() << ", line width=" << std::to_string(lineThicknessStack.top()) << "] ";
        myfile << "(" << std::to_string(imageScale*x1) << "," << std::to_string(imageScale*y1) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(imageScale*x2) << "," << std::to_string(imageScale*y2) << ");";
        myfile << std::endl;
    }

    void TikzObject::DrawGrid(double x1, double y1, double x2, double y2, int nx, int ny)
    {
        double dx = (x2-x1)/(nx);
        double dy = (y2-y1)/(ny);
        for (int i = 0; i < nx+1; i++)
        {
            DrawLine(x1+i*dx, y1, x1+i*dx, y2);
        }
        for (int j = 0; j < ny+1; j++)
        {
            DrawLine(x1, y1+j*dy, x2, y1+j*dy);
        }
    }

    void TikzObject::DrawBox(double x1, double y1, double x2, double y2)
    {
        DrawGrid(x1, y1, x2, y2, 1, 1);
    }
}
