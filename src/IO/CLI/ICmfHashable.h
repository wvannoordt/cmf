#ifndef CMF_HASHABLEOBJECT_H
#define CMF_HASHABLEOBJECT_H
#define PRIMEA 54059
#define PRIMEB 76963
#define PRIMEH 37
#include "CmfScreen.h"
namespace cmf
{
    /// @brief Provides an interface for objects to be hashed for parallel synchronization or IO
    /// @author WVN
    class ICmfHashable
    {
        public:
            /// @brief Empty constructor
            /// @author WVN
            ICmfHashable(void);
            
            /// @brief Empty destructor
            /// @author WVN
            ~ICmfHashable(void);
            
            /// @brief Returns the hash
            /// @author WVN
            size_t GetHash(void);
            
            /// @brief Augments the value of the current hash
            /// @param value The value to use for hash augmentation
            /// @author WVN
            void AugmentHash(int value);
            
            /// @brief Augments the value of the current hash
            /// @param value The value to use for hash augmentation
            /// @author WVN
            void AugmentHash(std::string value);
        
        private:
            /// @brief Empty destructor
            /// @author WVN
            size_t hash;
    };
}

#endif