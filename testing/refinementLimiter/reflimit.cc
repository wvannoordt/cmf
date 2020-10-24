#include <iostream>
#include <cmath>
#include "cmf.h"
bool limit(cmf::RefinementTreeNode* block)
{
    return (block->GetLevel() > 2);
}

int main(int argc, char** argv)
{
    __only2d
    (
        cmf::Initialize();
        cmf::ReadInput("input.ptl");
        cmf::CartesianMeshInputInfo inputInfo("Domain", cmf::mainInput);
        cmf::CartesianMesh domain(inputInfo);
        domain.Blocks()->SetRefineLimitCriterion(limit);
        double coords[2];
        double nvec[2];
        double tvec[2];
        int Ncirc = 1000;
        int Nrad = 1000;
        int Nline = 1000;
        double xc = 0.5;
        double yc = 0.5;
        double shockangle = 0.4;
        double shockradius = 0.2;
        double shocklen = 0.9;
        double rmean = 0.11;
        double rdelta = 0.04;
        for (int ri = 0; ri < Nrad; ri++)
        {
            for (int i = 0; i < Ncirc; i++)
            {
                double alpha = (double)(ri)/(Nrad-1);
                double r = (rmean-rdelta)*alpha + (rmean+rdelta)*(1.0-alpha);
                double theta = ((double)i-0.5)*6.28318530718/Ncirc;
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
                domain.Blocks()->RefineAt(coords, reftype);
            }
        }
        double beta = 1.57079632679 - shockangle;
        double thetaMin = 1.57079632679 + shockangle;
        double thetaMax = thetaMin + 2.0*beta;
        for (int i = 0; i < Nline; i++)
        {
            //circle
            double prog = ((double)i)/(Nline-1);
            double thetaCirc = thetaMin + (thetaMax-thetaMin)*prog;

            coords[0] = xc + shockradius*cos(thetaCirc);
            coords[1] = yc + shockradius*sin(thetaCirc);
            nvec[0] = cos(thetaCirc);
            nvec[1] = sin(thetaCirc);
            domain.Blocks()->RefineAt(coords, 3);

            //line, upper
            nvec[0] = cos(thetaMin);
            nvec[1] = sin(thetaMin);
            tvec[0] = cos(shockangle);
            tvec[1] = sin(shockangle);
            coords[0] = xc + shockradius*cos(thetaMin)+shocklen*prog*tvec[0];
            coords[1] = yc + shockradius*sin(thetaMin)+shocklen*prog*tvec[1];
            //domain.RefineAt(coords, 3);

            //line, lower
            nvec[0] = cos(thetaMax);
            nvec[1] = sin(thetaMax);
            tvec[0] = cos(-shockangle);
            tvec[1] = sin(-shockangle);
            coords[0] = xc + shockradius*cos(thetaMax)+shocklen*prog*tvec[0];
            coords[1] = yc + shockradius*sin(thetaMax)+shocklen*prog*tvec[1];
            domain.Blocks()->RefineAt(coords, 3);
        }
        std::string filename = "output/main.tex";
        cmf::TikzObject picture;
        picture.Open(filename);
    	picture.SetClip(0,0,1,1);
        domain.Blocks()->Render(&picture);
        //picture.FillCircle(xc, yc, rmean-0.001);
        picture.Close();
        cmf::Finalize();
    )
    return 0;
}
