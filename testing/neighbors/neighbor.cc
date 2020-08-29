#include <iostream>
#include <cmath>
#include "gTree.h"

int main(int argc, char** argv)
{
    gTree::Initialize();
    gTree::ReadInput("input.ptl");
    gTree::RefinementBlock domain("Domain");
    double xc = 0.55;
    double yc = 0.55;
    double coords[2];
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            coords[0] = xc + 0.1*(double)(i);
            coords[1] = yc + 0.1*(double)(j);
            if ((i!=0) || (j!= 0))
            {
                char ref = i*i + 2*j*j;
                domain.RefineAt(coords, ref);
            }
        }
    }
    std::string filename = "output/main.tex";
    gTree::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Render(&picture);
    picture.Close();
    gTree::Finalize();
    return 0;
}
