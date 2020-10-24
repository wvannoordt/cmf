#ifndef PNG_DRAW_H
#define PNG_DRAW_H
#include "PngImage.h"
namespace cmf
{
    /// @brief Class that draws simple shapes on a PngImage \see PngImage
	/// @author WVN
    class PngDrawer
    {
        public:
            /// @brief Constructor for the PngDrawer class
            /// @param image_in The PngImage to be drawn to
        	/// @author WVN
            PngDrawer(PngImage* image_in);
            
            /// @brief Destructor for PngDrawer class
        	/// @author WVN
            ~PngDrawer(void);
            
            /// @brief Fills the underlying image with a specified color
            /// @param color The color to fill
        	/// @author WVN
            void Fill(int color);
            
            /// @brief Fills the underlying image with a specified color
            /// @param color The color to fill
        	/// @author WVN
            void Fill(pxtype color);
            
            /// @brief Sets the coordinate system of the current canvas
            /// @param xmin_in x-coordinate corresponding to left boundary of image
            /// @param xmax_in x-coordinate corresponding to right boundary of image
            /// @param ymin_in y-coordinate corresponding to bottom boundary of image
            /// @param ymax_in y-coordinate corresponding to top boundary of image
        	/// @author WVN
            void SetCoordinateSystem(double xmin_in, double xmax_in, double ymin_in, double ymax_in);
            
            /// @brief Fills a box with the specified coordinates
            /// @param x0 x-coordinate corresponding to left boundary of box to fill
            /// @param x1 x-coordinate corresponding to right boundary of box to fill
            /// @param y0 y-coordinate corresponding to bottom boundary of box to fill
            /// @param y1 y-coordinate corresponding to top boundary of box to fill
            /// @param color The color to fill
        	/// @author WVN
            void FillBox(double x0, double x1, double y0, double y1, int color);
            
            /// @brief Fills a box with the specified coordinates
            /// @param x0 x-coordinate corresponding to left boundary of box to fill
            /// @param x1 x-coordinate corresponding to right boundary of box to fill
            /// @param y0 y-coordinate corresponding to bottom boundary of box to fill
            /// @param y1 y-coordinate corresponding to top boundary of box to fill
            /// @param color The color to fill
        	/// @author WVN
            void FillBox(double x0, double x1, double y0, double y1, pxtype color);
            
            /// @brief Fills a box with the specified coordinates, including a border
            /// @param x0 x-coordinate corresponding to left boundary of box to outline
            /// @param x1 x-coordinate corresponding to right boundary of box to outline
            /// @param y0 y-coordinate corresponding to bottom boundary of box to outline
            /// @param y1 y-coordinate corresponding to top boundary of box to outline
            /// @param fillColor The color to fill the box with
            /// @param borderColor The color to outline the box withincluding transparency
            /// @param borderWidth The thickness of the border in pixels
        	/// @author WVN
            void OutlineBox(double x0, double x1, double y0, double y1, pxtype fillColor, pxtype borderColor, int borderWidth);
            
            /// @brief Fills a box with the specified coordinates, including a border
            /// @param x0 x-coordinate corresponding to left boundary of box to outline
            /// @param x1 x-coordinate corresponding to right boundary of box to outline
            /// @param y0 y-coordinate corresponding to bottom boundary of box to outline
            /// @param y1 y-coordinate corresponding to top boundary of box to outline
            /// @param fillColor The color to fill the box with
            /// @param borderColor The color to outline the box withincluding transparency
            /// @param borderWidth The thickness of the border in pixels
        	/// @author WVN
            void OutlineBox(double x0, double x1, double y0, double y1, int fillColor, pxtype borderColor, int borderWidth);
            
            /// @brief Fills a box with the specified coordinates, including a border
            /// @param x0 x-coordinate corresponding to left boundary of box to outline
            /// @param x1 x-coordinate corresponding to right boundary of box to outline
            /// @param y0 y-coordinate corresponding to bottom boundary of box to outline
            /// @param y1 y-coordinate corresponding to top boundary of box to outline
            /// @param fillColor The color to fill the box with
            /// @param borderColor The color to outline the box withincluding transparency
            /// @param borderWidth The thickness of the border in pixels
        	/// @author WVN
            void OutlineBox(double x0, double x1, double y0, double y1, pxtype fillColor, int borderColor, int borderWidth);
            
            /// @brief Fills a box with the specified coordinates, including a border
            /// @param x0 x-coordinate corresponding to left boundary of box to outline
            /// @param x1 x-coordinate corresponding to right boundary of box to outline
            /// @param y0 y-coordinate corresponding to bottom boundary of box to outline
            /// @param y1 y-coordinate corresponding to top boundary of box to outline
            /// @param fillColor The color to fill the box with
            /// @param borderColor The color to outline the box withincluding transparency
            /// @param borderWidth The thickness of the border in pixels
        	/// @author WVN
            void OutlineBox(double x0, double x1, double y0, double y1, int fillColor, int borderColor, int borderWidth);
            
        private:
            
            /// @brief Transforms an x-y pair to an image coordinate pair
            void CoordsToIndices(double x, double y, int* i, int* j);
            
            /// @brief Class that draws simple shapes on a PngImage \see PngImage
            PngImage* image;
            
            /// @brief The width of the pixel buffer
            int width;
            
            /// @brief The height of the pixel buffer
            int height;
            
            /// @brief x-coordinate corresponding to left boundary of image
            double xmin;
            
            /// @brief x-coordinate corresponding to right boundary of image
            double xmax;
            
            /// @brief y-coordinate corresponding to bottom boundary of image
            double ymin;
            
            /// @brief y-coordinate corresponding to top boundary of image
            double ymax;
    };
}

#endif