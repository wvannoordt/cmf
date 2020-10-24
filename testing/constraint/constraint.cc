#include <iostream>
#include <cmath>
#include "cmf.h"
#define RX 0.542
#define RY 0.5231
static inline bool BoxContains(double* bounds, double* coords)
{
#if(IS3D)
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3])&&(coords[2]>bounds[4])&&(coords[2]<bounds[5]);
#else
    return (coords[0]>bounds[0])&&(coords[0]<bounds[1])&&(coords[1]>bounds[2])&&(coords[1]<bounds[3]);
#endif
}

void DebugDraw(cmf::TikzObject* picture, cmf::RefinementTreeNode* node)
{
    double rad = 0.006;
    double* blockBounds = node->GetBlockBounds();
    double x1 = 0.5*(blockBounds[0]+blockBounds[1])-rad;
    double y1 = 0.5*(blockBounds[2]+blockBounds[3])-rad;
    double x2 = 0.5*(blockBounds[0]+blockBounds[1])+rad;
    double y2 = 0.5*(blockBounds[2]+blockBounds[3])+rad;
    double xProbe[CMF_DIM];
    xProbe[0] = RX;
    xProbe[1] = RY;
    bool neighs = true;
    if (!neighs && node->IsAnyDomainBoundary())
    {
        picture->PushFillType(cmf::TikzColor::gray);
        picture->FillBox(blockBounds[0], blockBounds[2], blockBounds[1], blockBounds[3]);
        picture->PopFillType();
    }
    if (neighs && BoxContains(blockBounds, xProbe))
    {
        picture->PushFillType(cmf::TikzColor::teal);
        picture->FillBox(x1, y1, x2, y2);
        picture->PushFillType(cmf::TikzColor::green);
        picture->FillBox(xProbe[0]-rad, xProbe[1]-rad, xProbe[0]+rad, xProbe[1]+rad);
        picture->PushFillType(cmf::TikzColor::red);
        for (cmf::NeighborIterator it(node); it.Active(); it++)
        {

            cmf::RefinementTreeNode* n = it.Node();
            cmf::NodeEdge edge = it.Edge();
            double* nBlockBounds = n->GetBlockBounds();
            double xm  = 0.5*(nBlockBounds[0]+nBlockBounds[1]);
            double ym  = 0.5*(nBlockBounds[2]+nBlockBounds[3]);
            double x1n = 0.5*(nBlockBounds[0]+nBlockBounds[1])-rad;
            double y1n = 0.5*(nBlockBounds[2]+nBlockBounds[3])-rad;
            double x2n = 0.5*(nBlockBounds[0]+nBlockBounds[1])+rad;
            double y2n = 0.5*(nBlockBounds[2]+nBlockBounds[3])+rad;
            double dx = 0.9*rad;
            picture->FillBox(x1n, y1n, x2n, y2n);
            picture->DrawLine(xm, ym, xm-edge.edgeVector[0]*dx, ym-edge.edgeVector[1]*dx);
        }
        picture->PopFillType();
        picture->PopFillType();
        picture->PopFillType();
    }
}

int main(int argc, char** argv)
{
    if (CMF_DIM != cmf::GetDim())
    {
        cmf::cmfout << "WARNING: skipping test case in file " << __FILE__ << ": dimensions incompatible." << cmf::cmfendl;
        return 0;
    }
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::CartesianMeshInputInfo inputInfoA("DomainA", cmf::mainInput);
    cmf::CartesianMeshInputInfo inputInfoB("DomainB", cmf::mainInput);
    cmf::CartesianMesh domainA(inputInfoA);
    cmf::CartesianMesh domainB(inputInfoB);

    cmf::CartesianMesh* domains[2];
    domains[0] = &domainA;
    domains[1] = &domainB;


    for (int i = 0; i < 2; i++)
    {
        double coords[2];
        coords[0] = RX+i*1.1;
        coords[1] = RY;
        domains[i]->Blocks()->RefineRandom();
        domains[i]->Blocks()->RefineRandom();
        domains[i]->Blocks()->RefineRandom();
    }

    std::string filename = "output/main.tex";
    cmf::TikzObject picture;
    picture.Open(filename);
    cmf::DebugTikzDraw_t neighborDraw(DebugDraw);
    for (int i = 0; i < 2; i++)
    {
        domains[i]->Blocks()->Render(&picture, neighborDraw);
    }
    picture.Close();
    cmf::Finalize();
    return 0;
}
