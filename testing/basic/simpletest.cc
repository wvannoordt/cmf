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

        std::string filename = "output/main.tex";
        Anaptric::TikzObject picture;
        picture.Open(filename);
        domain.Render(&picture);
        picture.Close();
        Anaptric::Finalize();
    )
    return 0;
}
