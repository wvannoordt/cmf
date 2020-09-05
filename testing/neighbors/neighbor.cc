#include <iostream>
#include <cmath>
#include "Anaptric.h"

int main(int argc, char** argv)
{
    __only2d
    (
        Anaptric::Initialize();
        Anaptric::ReadInput("input.ptl");
        Anaptric::RefinementBlock domain("Domain");
        double coords[2];
        coords[0] = 0.5;
        coords[1] = 0.6;
        domain.RefineRandom();
        domain.RefineRandom();
        domain.RefineRandom();
        domain.RefineRandom();
        //domain.RefineRandom();
        //domain.RefineAt(coords, 2);
        std::string filename = "output/main.tex";
        Anaptric::TikzObject picture;
        picture.Open(filename);
    	picture.SetClip(0,0,1,1);
        domain.Render(&picture);
        picture.Close();
        Anaptric::Finalize();
    )
    return 0;
}
