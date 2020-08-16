#ifndef TikzObject_H
#define TikzObject_H

#include <string>
#include "PropTreeLib.h"
#include <iostream>
#include <fstream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "TikzLineType.h"
#include "TikzColor.h"
#include <stack>

namespace gTree
{
    class TikzObject
    {
        public:
            TikzObject(void);
            ~TikzObject(void);
            void Open(std::string filename);
            
            void DrawLine(double x1, double y1, double x2, double y2);
            void DrawGrid(double x1, double y1, double x2, double y2, int nx, int ny);
            void DrawBox(double x1, double y1, double x2, double y2);
            void FillBox(double x1, double y1, double x2, double y2);
            void FillCircle(double x1, double y1, double r);
            void DrawCircle(double x1, double y1, double r);

            void PushLineType(TikzLineType::TikzLineType style, double thickness, TikzColor::TikzColor color);
            void PushLineType(TikzLineType::TikzLineType style, double thickness, std::string color);
            void PushLineType(std::string style, double thickness, TikzColor::TikzColor color);
            void PushLineType(std::string style, double thickness, std::string color);
            void PopLineType(void);
            
            void PushFillType(TikzColor::TikzColor color);
            void PushFillType(std::string color);
            void PopFillType(void);

            void Close(void);
        private:
            double CoordR(double r);
            double CoordX(double x);
            double CoordY(double y);
            std::ofstream myfile;
            double imageScale;
            std::stack<std::string> lineStyleStack;
            std::stack<std::string> lineColorStack;
            std::stack<double> lineThicknessStack;
            
            std::stack<std::string> fillColorStack;

    };
}

#endif
