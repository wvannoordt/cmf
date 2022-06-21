#include "CartesianMeshInterpolationOperator.h"
#include "StringUtils.h"
#include "CmfScreen.h"
#include "Utils.hx"
namespace cmf
{
    CartesianMeshInterpolationOperator::CartesianMeshInterpolationOperator(CartesianMesh& source_in, CartesianMesh& destination_in)
    {
        source = &source_in;
        destination = &destination_in;
        dataDonors.clear();
        for (auto recv: destination_in)
        {
            dataDonors.insert({recv, std::vector<RefinementTreeNode*>()});
            for (auto sender: source_in)
            {
                auto recvbox = recv->GetBlockBoundsVec6();
                auto sendbox = sender->GetBlockBoundsVec6();
                if (BoxOverlap3(&(recvbox.v[0]), &(sendbox.v[0])))
                {
                    dataDonors[recv].push_back(sender);
                }
            }
        }
    }
    
    void CartesianMeshInterpolationOperator::Interpolate(CartesianMeshArray& sourceArray, CartesianMeshArray& destinationArray)
    {
        std::string frmt = "Interpolating variable \"{}\" to variable \"{}\"";
        WriteLine(3, strformat(frmt, sourceArray.GetFullName(), destinationArray.GetFullName()));
        
    }
}