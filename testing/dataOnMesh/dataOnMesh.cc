#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::globalSettings = cmf::GlobalSettings(cmf::mainInput["GlobalSettings"]);
    cmf::CartesianMeshInputInfo inputInfo(cmf::mainInput["Domain"]);
    cmf::CartesianMesh domain(inputInfo);
    double coords[2];
    coords[0] = 0.23;
    coords[1] = 0.54;
    double radius = 0.1;
    cmf::AxisAlignedLongCylinder cyl(coords, radius, 2);
    domain.Blocks()->RefineRandom();
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > 2);});

    for (cmf::BlockIterator lb(domain.Blocks(), cmf::BlockFilters::Terminal); lb.HasNext(); lb++)
    {
        if (cyl.BoxIntersectsBoundary(lb.Node()->GetBlockBounds())) lb.Node()->Refine(7);
    }
    domain.Blocks()->RefineRandom();
    
    
    
    cmf::CartesianMeshArray dist = *(domain.DefineVariable("distance"));
    int* meshDims = inputInfo.meshDataDim;
    double xi, yi, dx, dy;
    double cellBoundBox[4];
    //compute data
    double dmin, dmax;
    dmin = 1e30;
    dmax = -1e30;
    for (cmf::BlockIterator lb(&dist); lb.HasNext(); lb++)
    {
        double* distData = (double*)dist[lb];
        double* bounds = lb.Node()->GetBlockBounds();
        dx = (bounds[1]-bounds[0])/meshDims[0];
        dy = (bounds[3]-bounds[2])/meshDims[1];
        for (int j = 0; j < meshDims[1]; j++)
        {
            for (int i = 0; i < meshDims[0]; i++)
            {
                xi = bounds[0]+((double)i+0.5)*dx;
                yi = bounds[2]+((double)j+0.5)*dy;
                int index = i + j*meshDims[0];
                distData[index] = sqrt((xi-coords[0])*(xi-coords[0]) + (yi-coords[1])*(yi-coords[1])) - 0.1;
                dmin = (dmin<distData[index])?(dmin):(distData[index]);
                dmax = (dmax>distData[index])?(dmax):(distData[index]);
            }
        }
    }
    cmf::PngImage img(2048,2048);
    cmf::PngDrawer canvas(&img);
    double* boundsL = inputInfo.blockBounds;
    canvas.SetCoordinateSystem(boundsL[0], boundsL[1], boundsL[2], boundsL[3]);
    cmf::pxtype black = {0, 0, 0, 255};
    cmf::pxtype gray = {45, 45, 45, 255};
    cmf::pxtype t_red = {0, 0, 255, 145};
    cmf::pxtype white = {255, 255, 255, 255};
    cmf::pxtype green = {0, 255, 0, 255};
    cmf::pxtype col = {0, 0, 0, 0};
    
    canvas.Fill(black);
    int num = 0;
    for (cmf::BlockIterator lb(&dist); lb.HasNext(); lb++)
    {
        double* distData = (double*)dist[lb];
        double* bounds = lb.Node()->GetBlockBounds();
        dx = (bounds[1]-bounds[0])/meshDims[0];
        dy = (bounds[3]-bounds[2])/meshDims[1];
        for (int j = 0; j < meshDims[1]; j++)
        {
            for (int i = 0; i < meshDims[0]; i++)
            {
                xi = bounds[0]+((double)i+0.5)*dx;
                yi = bounds[2]+((double)j+0.5)*dy;
                cellBoundBox[0] = xi-0.5*dx;
                cellBoundBox[1] = xi+0.5*dx;
                cellBoundBox[2] = yi-0.5*dy;
                cellBoundBox[3] = yi+0.5*dy;
                int index = i + j*meshDims[0];
                
                double zz = (distData[index]-dmin) / (dmax - dmin);
                col[0] = (unsigned char)255.0*(zz);
                col[1] = 0;
                col[2] = (unsigned char)255.0*(1.0-zz);
                col[3] = 255;
                canvas.OutlineBox(cellBoundBox[0], cellBoundBox[1], cellBoundBox[2], cellBoundBox[3], col, black, 1);
            }
        }
        canvas.OutlineBox(bounds[0], bounds[1], bounds[2], bounds[3], 0, green, 1);
        num++;
    }
    img.Write("output/domain.png");
    cmf::Finalize();
    return 0;
}
