#include <iostream>
#include <cmath>
#include "cmf.h"
#include "cmftestutils.h"
double pldist(double* c, double x1, double y1, double x2, double y2, double r)
{
    double x0 = c[0];
    double y0 = c[1];
    double a = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1;
    if (a<0) a = -a;
    double d = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
    double alpha = (x1-x2)*(x1-x0) + (y1-y2)*(y1-y0);
    return (a/d < r) && ((sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))>r)||(sqrt((x2-x0)*(x2-x0) + (y2-y0)*(y2-y0))>r)) && (0 <= alpha) && (alpha <= d*d);
}
bool circleIntersects(double* c, double r, double* box, bool dooutput)
{
    bool negativeFound = false;
    bool positiveFound = false;
    double corner[2];
    for (int i = 0; i < 4; i++)
    {
        corner[0] = (i&1)*box[0] + (1 - (i&1))*box[1];
        corner[1] = ((i&2)/2)*box[2] + (1-((i&2)/2))*box[3];
        double q = (c[0]-corner[0])*(c[0]-corner[0]) + (c[1]-corner[1])*(c[1]-corner[1]) - r*r;
        negativeFound = negativeFound || (q<0);
        positiveFound = positiveFound || (q>0);
    }
    bool aCornerIsIn = (positiveFound && negativeFound);
    bool isContained = true;
    isContained = isContained && ((c[0] - r) > box[0]);
    isContained = isContained && ((c[0] + r) < box[1]);
    isContained = isContained && ((c[1] - r) > box[2]);
    isContained = isContained && ((c[1] + r) < box[3]);
    bool anEdgeIsIn = false;
    anEdgeIsIn = anEdgeIsIn || (pldist(c, box[0], box[2], box[0], box[3], r));
    anEdgeIsIn = anEdgeIsIn || (pldist(c, box[1], box[2], box[1], box[3], r));
    anEdgeIsIn = anEdgeIsIn || (pldist(c, box[0], box[2], box[1], box[2], r));
    anEdgeIsIn = anEdgeIsIn || (pldist(c, box[0], box[3], box[1], box[3], r));
    //return aCornerIsIn;
    //return aCornerIsIn;
    return aCornerIsIn || isContained || anEdgeIsIn;
}
double r = 0.01;
double coords[2];
bool circGeom(cmf::RefinementTreeNode* n)
{
    bool output = circleIntersects(coords, r, n->GetBlockBounds(), false);
    //std::cout << n->GetLevel() << "->" << output << std::endl;
    return output;
}
bool circGeomNoisy(cmf::RefinementTreeNode* n)
{
    bool output = circleIntersects(coords, r, n->GetBlockBounds(), true);
    //std::cout << n->GetLevel() << "->" << output << std::endl;
    return output;
}
int main(int argc, char** argv)
{
    EXIT_WARN_IF_DIM_NOT(2);
    cmf::Initialize();
    cmf::ReadInput("input.ptl");
    cmf::cmfout << "BEFORE" << cmf::cmfendl;
    cmf::CartesianMeshInputInfo inputInfo("Domain", cmf::mainInput);
    cmf::cmfout << "AFTER" << cmf::cmfendl;
    cmf::CartesianMesh domain(inputInfo);
    //coords[0] = 0.11;
    //coords[1] = 0.52;
    //double r = 0.3;
    coords[0] = 0.105;
    coords[1] = 0.54;
    //domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->MaxEdgeLength() < 0.03) && (n->GetLevel() > 4);});
    domain.Blocks()->SetRefineLimitCriterion([](cmf::RefinementTreeNode* n){return (n->GetLevel() > 4);});
    cmf::NodeFilter_t circle(circGeom);
    cmf::NodeFilter_t circle2(circGeomNoisy);
    int c = 0;
    for (cmf::BlockIterator i(domain.Blocks(), circle); i.HasNext(); i++)
    {
        //i.Node()->Refine((char)(1+(c++)%3));
        i.Node()->Refine(7);
        //std::cout << i.Node()->GetLevel() << std::endl;
    }
    int p = 0;
    for (cmf::BlockIterator i(domain.Blocks(), [](cmf::RefinementTreeNode* n){return (n->IsTerminal());}); i.HasNext(); i++)
    {
        std::cout << "BLOCK " << p++ << " " << circle2(i.Node()) << std::endl;
    }
    std::string filename = "output/main.tex";
    cmf::TikzObject picture;
    picture.Open(filename);
	picture.SetClip(0,0,1,1);
    domain.Blocks()->Render(&picture);
    picture.PushLineType("solid", 0.03, "red");
    picture.DrawCircle(coords[0], coords[1], r);
    picture.Close();
    cmf::Finalize();
    return 0;
}
