#ifndef ICMF_HAS_BLOCK_BOUND_INDICES_H
#define ICMF_HAS_BLOCK_BOUND_INDICES_H

namespace cmf
{
    /// @brief A struct representing an object that contains some imin, imax, jmin, jmax, etc.
    /// Used to prevent code duplication between BlockArray and BlockInfo
    /// @author WVN
    struct ICmfHasBlockBoundIndices
    {
        /// @brief minimum i index
        int imin;
        /// @brief maximum i index
        int imax;
        /// @brief minimum j index
        int jmin;
        /// @brief maximum j index
        int jmax;
        /// @brief minimum k index
        int kmin;
        /// @brief maximum k index
        int kmax;
        /// @brief number of exchange cells in i-direction
        int exchangeI;
        /// @brief number of exchange cells in j-direction
        int exchangeJ;
        /// @brief number of exchange cells in k-direction
        int exchangeK;
    };
}

#endif