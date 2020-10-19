#include <iostream>
#include <cmath>
#include "cmf.h"

int main(int argc, char** argv)
{
    __only2d
    (
        cmf::Initialize();
        cmf::ReadInput("input.ptl");
        cmf::CartesianMesh domain("Domain");
        double coords[2];
        coords[0] = 0.5;
        coords[1] = 0.6;
        domain.Blocks()->RefineRandom();
        domain.Blocks()->RefineRandom();
        domain.Blocks()->RefineRandom();
        domain.Blocks()->RefineRandom();
        //domain.RefineRandom();
        //domain.RefineAt(coords, 2);
        std::string filename = "output/main.tex";
        cmf::TikzObject picture;
        picture.Open(filename);
    	picture.SetClip(0,0,1,1);
        domain.Blocks()->Render(&picture);
        picture.Close();
        cmf::Finalize();
    )
    return 0;
}
