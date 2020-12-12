#ifndef TikzObject_H
#define TikzObject_H

#include <string>
#include "PTL.h"
#include <iostream>
#include <fstream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "TikzLineType.h"
#include "TikzColor.h"
#include <stack>

namespace cmf
{
    /// @brief Class for making simple TikZ figures, useful for debugging. Similar usage as a std::ofstream,
    /// having Open(filename) and Close()
	/// @author WVN
    class TikzObject
    {
        public:
            /// @brief Create a TikzObject
            /// @author WVN
            TikzObject(void);
            
            /// @brief Destroy the TikzObject
            /// @author WVN
            ~TikzObject(void);
            
            /// @brief Opens the underlying stream with filename and write the latex header.
    		/// @param filename  File to output LaTex/TikZ code to
            /// @author WVN
            void Open(std::string filename);
            
            /// @brief Draws a line to the TikZ picture using the current line style.
            /// @param x1  x-coordinate of first point
            /// @param y1  y-coordinate of first point
            /// @param x2  x-coordinate of second point
            /// @param y2  y-coordinate of second point
            /// @author WVN
            void DrawLine(double x1, double y1, double x2, double y2);
            
            /// @brief Draws an arrow to the TikZ picture using the current line style.
            /// @param x1  x-coordinate of first point
            /// @param y1  y-coordinate of first point
            /// @param x2  x-coordinate of second point
            /// @param y2  y-coordinate of second point
            /// @author WVN
            void DrawArrow(double x1, double y1, double x2, double y2);
            
            /// @brief Draws a grid of lines to the TikZ picture using the current line style.
            /// @pre The points defined by (x1,y1) and (x2,y2) should be opposite corners of the grid.
            /// @param x1  x-coordinate of first corner
            /// @param y1  y-coordinate of first corner
            /// @param x2  x-coordinate of second corner
            /// @param y2  y-coordinate of second corner
            /// @param nx  Number of lines in the x-direction
            /// @param ny  Number of lines in the y-direction
            /// @author WVN
            void DrawGrid(double x1, double y1, double x2, double y2, int nx, int ny);
            
            /// @brief Draws a box to the TikZ picture using the current line style.
            /// @pre The points defined by (x1,y1) and (x2,y2) should be opposite corners of the box.
            /// @param x1  x-coordinate of first corner
            /// @param y1  y-coordinate of first corner
            /// @param x2  x-coordinate of second corner
            /// @param y2  y-coordinate of second corner
            /// @author WVN
            void DrawBox(double x1, double y1, double x2, double y2);
            
            /// @brief Fills a box on the TikZ picture using the current fill style.
            /// @pre The points defined by (x1,y1) and (x2,y2) should be opposite corners of the box.
            /// @param x1  x-coordinate of first corner
            /// @param y1  y-coordinate of first corner
            /// @param x2  x-coordinate of second corner
            /// @param y2  y-coordinate of second corner
            /// @author WVN
            void FillBox(double x1, double y1, double x2, double y2);
            
            /// @brief Fills a circle on the TikZ picture using the current fill style.
            /// @param x1  x-coordinate of the center
            /// @param y1  y-coordinate of the center
            /// @param r  radius
            /// @author WVN
            void FillCircle(double x1, double y1, double r);
            
            /// @brief Draws a circle to the TikZ picture using the current fill style.
            /// @param x1  x-coordinate of the center
            /// @param y1  y-coordinate of the center
            /// @param r  radius
            /// @author WVN
            void DrawCircle(double x1, double y1, double r);
            
            /// @brief Restricts the dimensions of a picture to the box defined by (x1,y1) and (x2,y2).
            /// @pre The points defined by (x1,y1) and (x2,y2) should be opposite corners of the box.
            /// @param x1  x-coordinate of first corner
            /// @param y1  y-coordinate of first corner
            /// @param x2  x-coordinate of second corner
            /// @param y2  y-coordinate of second corner
            /// @author WVN
            void SetClip(double x1, double y1, double x2, double y2);
            
            /// @brief Pushes a line style to the line style stack. \see TikzLineType.h TikzColor.h
            /// @param style  a TikzLineType enum defining a line style
            /// @param thickness  Thickness of the line
            /// @param color  a TikzColor enum specifying the line color
            /// @author WVN
            void PushLineType(TikzLineType::TikzLineType style, double thickness, TikzColor::TikzColor color);
            /// @brief Pushes a line style to the line style stack. \see TikzLineType.h
            /// @param style  a TikzLineType enum defining a line style
            /// @param thickness  Thickness of the line
            /// @param color  a Tikz-style string (e.g. "red!40!white") specifying the line color
            /// @author WVN
            void PushLineType(TikzLineType::TikzLineType style, double thickness, std::string color);
            
            /// @brief Pushes a line style to the line style stack. \see TikzColor.h
            /// @param style  a Tikz-style string (e.g. "dash pattern={on 7pt off 2pt on 1pt off 3pt}") defining a line style
            /// @param thickness  Thickness of the line
            /// @param color  a TikzColor enum specifying the line color
            /// @author WVN
            void PushLineType(std::string style, double thickness, TikzColor::TikzColor color);
            
            /// @brief Pushes a line style to the line style stack. \see TikzColor.h
            /// @param style  a Tikz-style string (e.g. "dash pattern={on 7pt off 2pt on 1pt off 3pt}") defining a line style
            /// @param thickness  Thickness of the line
            /// @param color  a Tikz-style string (e.g. "red!40!white") specifying the line color
            /// @author WVN
            void PushLineType(std::string style, double thickness, std::string color);
            
            /// @brief Pops a line style off of the line style stack.
            /// @author WVN
            void PopLineType(void);

            /// @brief Pushes a fill style to the fill style stack. \see TikzColor.h
            /// @param color  a TikzColor enum specifying the line color
            /// @author WVN
            void PushFillType(TikzColor::TikzColor color);
            
            /// @brief Pushes a fill style to the fill style stack.
            /// @param color  a Tikz-style string (e.g. "red!40!white") specifying the line color
            /// @author WVN
            void PushFillType(std::string color);
            
            /// @brief Pops a fill style off of the fill style stack.
            /// @author WVN
            void PopFillType(void);

            /// @brief Writes a TikZ footer and closes underlying stream.
            /// @author WVN
            void Close(void);
        private:
            /// @brief Used to apply a general coordinate transformation in cylindrical coordinates.
            /// @author WVN
            double CoordR(double r);
            
            /// @brief Used to apply a general coordinate transformation in Cartesian coordinates (x-direction).
            /// @author WVN
            double CoordX(double x);
            
            /// @brief Used to apply a general coordinate transformation in Cartesian coordinates (y-direction).
            /// @author WVN
            double CoordY(double y);
            
            /// @brief Underlying stream object
            std::ofstream myfile;
            
            /// @brief Used to scale the coordinates of the image \see CoordX CoordY CoordR
            double imageScale;
            
            /// @brief Stack for the line styles
            std::stack<std::string> lineStyleStack;
            
            /// @brief Stack for the line colors
            std::stack<std::string> lineColorStack;
            
            /// @brief Stack for the line thicknesses
            std::stack<double> lineThicknessStack;

            /// @brief Stack for the fill colors
            std::stack<std::string> fillColorStack;

    };
}

#endif
