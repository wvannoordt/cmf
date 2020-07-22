#ifndef TikzObject_H
#define TikzObject_H

#include <string>
#include "PropTreeLib.h"
#include "RefinementTreeNode.h"
#include <iostream>
#include <fstream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "TikzLineType.h"
#include "TikzColor.h"

namespace gTree
{
    class TikzObject
    {
        public:
            TikzObject(void);
            ~TikzObject(void);
            void Open(std::string filename);

            void DrawLine(double x1, double y1, double x2, double y2);
            void DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor);
            void DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor, double lineThickness);
            void DrawLine(double x1, double y1, double x2, double y2, std::string lineColor, double lineThickness, TikzLineType::TikzLineType lineType);
            void DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor, double lineThickness, std::string lineType);
            void DrawLine(double x1, double y1, double x2, double y2, TikzColor::TikzColor lineColor, double lineThickness, TikzLineType::TikzLineType lineType);
            void DrawLine(double x1, double y1, double x2, double y2, std::string lineColor, double lineThickness, std::string lineType);

            void Close(void);
        private:
            std::ofstream myfile;
            double imageScale;

    };
}

#endif
