#include <iostream>
#include <cmath>
#include "gTree.h"
bool limit(gTree::RefinementTreeNode* block)
{
    return (block->GetLevel() > 2);
}

int main(int argc, char** argv)
{
    gTree::Initialize();    
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    domain.SetRefineLimitCriterion(limit);
    double coords[2];
    int N1 = 5000;
    int N2 = 1000;
    double xc = 0.5;
    double yc = 0.5;
    double rmean = 0.3;
    double rdelta = 0.15;
    for (int ri = 0; ri < N2; ri++)
    {
        for (int i = 0; i < N1; i++)
        {
            double alpha = (double)(ri)/(N2-1);
            double r = (rmean-rdelta)*alpha + (rmean+rdelta)*(1.0-alpha);
            double theta = i*6.28318530718/N1;
            coords[0] = xc + r*cos(theta);
            coords[1] = yc + r*sin(theta);
            char reftype = 3;
            double q = theta/6.28318530718;
            double lim = 0.008;
            if ((q-0.25)*(q-0.25) < lim) reftype = 2;
            if ((q-0.75)*(q-0.75) < lim) reftype = 2;
            if ((q-0.5)*(q-0.5) < lim) reftype = 1;
            if ((q-0.0)*(q-0.0) < lim) reftype = 1;
            if ((q-1.0)*(q-1.0) < lim) reftype = 1;
            domain.RefineAt(coords, reftype);
        }
    }
    
    std::string filename = "output/main.tex";
    gTree::TikzObject picture;
    picture.Open(filename);
    domain.Render(&picture);
    picture.FillCircle(xc, yc, rmean-0.01);
    picture.Close();
    gTree::Finalize();
    return 0;
}
