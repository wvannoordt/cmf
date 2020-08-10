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
        PushLineType(TikzLineType::solid, 0.5, TikzColor::black);
        PushFillType(TikzColor::lightgray);
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
    
    void TikzObject::PushFillType(TikzColor::TikzColor color)
    {PushFillType(TikzColorStr(color));}
    
    void TikzObject::PushFillType(std::string color)
    {
        fillColorStack.push(color);
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

    void TikzObject::PopFillType(void)
    {
        if (fillColorStack.size()>1)
        {
            fillColorStack.pop();
        }
    }

    void TikzObject::DrawLine(double x1, double y1, double x2, double y2)
    {
        myfile << "\\draw[" << lineColorStack.top() << ", " << lineStyleStack.top() << ", line width=" << std::to_string(lineThicknessStack.top()) << "] ";
        myfile << "(" << std::to_string(CoordX(x1)) << "," << std::to_string(CoordY(y1)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x2)) << "," << std::to_string(CoordY(y2)) << ");";
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
        myfile << "\\draw[" << lineColorStack.top() << ", " << lineStyleStack.top() << ", line width=" << std::to_string(lineThicknessStack.top()) << "] ";
        myfile << "(" << std::to_string(CoordX(x1)) << "," << std::to_string(CoordY(y1)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x2)) << "," << std::to_string(CoordY(y1)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x2)) << "," << std::to_string(CoordY(y2)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x1)) << "," << std::to_string(CoordY(y2)) << ")";
        myfile << " -- ";
        myfile << "cycle;";
        myfile << std::endl;
    }
    
    void TikzObject::FillBox(double x1, double y1, double x2, double y2)
    {
        myfile << "\\draw[fill=" << fillColorStack.top() << ", line width=0.0] ";
        myfile << "(" << std::to_string(CoordX(x1)) << "," << std::to_string(CoordY(y1)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x2)) << "," << std::to_string(CoordY(y1)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x2)) << "," << std::to_string(CoordY(y2)) << ")";
        myfile << " -- ";
        myfile << "(" << std::to_string(CoordX(x1)) << "," << std::to_string(CoordY(y2)) << ")";
        myfile << " -- ";
        myfile << "cycle;";
        myfile << std::endl;
    }
    
    double TikzObject::CoordX(double x)
    {
        return imageScale*x;
    }
    
    double TikzObject::CoordY(double y)
    {
        return imageScale*y;
    }
}
