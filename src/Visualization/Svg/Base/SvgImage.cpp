#include "SvgImage.h"
#include "CmfError.h"
#include "SvgFileStream.h"
#include "SvgAttributes.h"
#include "SvgRectangle.h"
#include <cmath>
namespace cmf
{
    SvgImage::SvgImage(void)
    {
        Build(0,1,0,1);
    }
    
    SvgImage::SvgImage(double xmin, double xmax, double ymin, double ymax)
    {
        Build(xmin, xmax, ymin, ymax);
    }
    
    void SvgImage::Build(double xmin, double xmax, double ymin, double ymax)
    {
        ImageTransformation def;
        defaultGroup = CreateGroup("cmf_svg_default", true);
        BindGroup(defaultGroup);
        BindImage(this);
        SetBounds(xmin, xmax, ymin, ymax);
        hasFillColor = false;
    }
    
    void SvgImage::SetBounds(double xmin, double xmax, double ymin, double ymax)
    {
        bounds[0] = xmin;
        bounds[1] = xmax;
        bounds[2] = ymin;
        bounds[3] = ymax;
    }
    
    void SvgImage::SetFillColor(std::string color)
    {
        hasFillColor = true;
        fillColor = color;
    }
    
    std::vector<SvgElementGroup*>& SvgImage::GetGroups(void)
    {
        return elementGroups;
    }
    
    std::vector<std::string>& SvgImage::GetGroupNames(void)
    {
        return elementNames;
    }
    
    std::map<std::string, int>& SvgImage::GetGroupTable(void)
    {
        return elementLocations;
    }
    
    void SvgImage::Write(std::string filename)
    {
        SvgFileStream stream;
        stream.Open(filename);
        SvgAttributes meta;
        meta.Add("version", "1.0");
        meta.Add("encoding", "UTF-8");
        meta.Add("standalone", "no");
        stream.AddLine("<?xml " + meta.GetString() + "?>");
        SvgAttributes svgHeader;
        svgHeader.Add("id", "svg8");
        svgHeader.Add("version", "1.1");
        svgHeader.Add("viewBox", std::to_string(bounds[0]) + " " + std::to_string(bounds[2]) + " " + std::to_string(bounds[1] - bounds[0]) + " " + std::to_string(bounds[3] - bounds[2]));
        stream.AddLine("<svg " + svgHeader.GetString() + ">");
        stream.Indent();
        if (hasFillColor)
        {
            double wid = abs(bounds[1]-bounds[0]);
            double hei = abs(bounds[3]-bounds[2]);
            SvgNode lowerLeft(bounds[0]-0.3*wid, bounds[2]-0.3*hei);
            SvgNode upperRight(bounds[1]+0.3*wid, bounds[3]+0.3*hei);
            SvgRectangle baseRect(&lowerLeft, &upperRight, this);
            baseRect.SetFillColor(fillColor.StringValue());
            baseRect.WriteToFile(stream);
        }
        for (int i = 0; i < elementGroups.size(); i++)
        {
            if (elementGroups[i]->IsVisible())
            {
                elementGroups[i]->WriteToFile(stream);
            }
        }
        stream.UnIndent();
        stream.AddLine("</svg>");
        stream.Close();
    }
    
    SvgElementGroup& SvgImage::operator [] (std::string name)
    {
        if (!HasGroup(name))
        {
            return *CreateGroup(name);
        }
        else
        {
            return *GetGroup(name);
        }
    }
    
    double SvgImage::MapPoint(double xin, double yin, double* xout, double* yout)
    {
        double youtTemp = bounds[3] - (yin - bounds[2]);
        double xoutTemp = xin;
        for (int i = 0; i < transforms.size(); i++)
        {
            double xx = transforms[i].m11*xoutTemp + transforms[i].m12*youtTemp + transforms[i].b1;
            double yy = transforms[i].m21*xoutTemp + transforms[i].m22*youtTemp + transforms[i].b2;
            xoutTemp = xx;
            youtTemp = yy;
        }
        *yout = youtTemp;
        *xout = xoutTemp;
    }
    
    bool SvgImage::HasGroup(std::string name)
    {
        return (elementLocations.find(name) != elementLocations.end());
    }
    
    SvgElementGroup* SvgImage::CreateGroup(std::string name)
    {
        return CreateGroup(name, false);
    }
    
    SvgElementGroup* SvgImage::CreateGroup(std::string name, bool ignoreReserved)
    {
        if (NameIsReserved(name)&&!ignoreReserved) CmfError("SvgImage attempted to create group with name \"" + name + "\", which is reserved for internal use");
        if (HasGroup(name)) CmfError("SvgImage attempted to create duplicate group \"" + name + "\"");
        SvgElementGroup* newGroup = new SvgElementGroup(name, this);
        newGroup->SetPosition(elementGroups.size());
        elementLocations.insert({name, elementGroups.size()});
        elementGroups.push_back(newGroup);
        elementNames.push_back(name);
        return newGroup;
    }
    
    bool SvgImage::NameIsReserved(std::string name)
    {
        if (name == "cmf_svg_default") return true;
        return false;
    }
    
    SvgElementGroup* SvgImage::GetGroup(std::string name)
    {
        if (!HasGroup(name)) CmfError("SvgImage attempted to access nonexistent group \"" + name + "\"");
        return elementGroups[elementLocations[name]];
    }
    
    SvgImage::~SvgImage(void)
    {
        for (int i = 0; i < elementGroups.size(); i++)
        {
            delete (elementGroups[i]);
        }
    }
}