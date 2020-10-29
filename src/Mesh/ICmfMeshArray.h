#ifndef ICMF_MESH_ARRAY_H
#define ICMF_MESH_ARRAY_H
#include <string>
#include "ArrayInfo.h"
#include "CmfError.h"
namespace cmf
{
    /// @brief Defines a general MeshArray object for various grid types
    /// @author WVN
    class ICmfMeshArray
    {
        public:
            /// @brief Copy constructor
            /// @param info Array information
            /// @param mesh_in The mesh over which this array is defined
            /// @author WVN
            ICmfMeshArray(ArrayInfo info)
            {
                variableName = info.name;
                if (info.rank > MAX_RANK) CmfError("Rank of variable \"" + variableName + "\" exceeds MAX_RANK (" + std::to_string(MAX_RANK) + "): recompile with greater limit.");
                for (int i = 0; i < info.rank; i++)
                {
                    dims[i] = info.dimensions[i];
                }
            }

            /// @brief Explicitly release resources used by this array
            /// @author WVN
            virtual void Destroy(void){}

            /// @brief Empty destructor
            /// @author WVN
            ~ICmfMeshArray(void){}

        protected:

            /// @brief The name of the variable this array represents
            std::string variableName;

            /// @brief The rank of this array
            int rank;

            /// @brief The dimensions of this array
            int dims[MAX_RANK];
    };
}

#endif
