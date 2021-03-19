#ifndef ICMF_FILE_READ_WRITE_OBJECT_H
#define ICMF_FILE_READ_WRITE_OBJECT_H

namespace cmf
{
    /// @brief Represents an object that can read from and written to a CMF data file
    /// @author WVN
    class ICmfDataFileReadWriteObject
    {
        public:
            /// @brief Constructor
            ICmfDataFileReadWriteObject(void);
            /// @brief Destructor
            virtual ~ICmfDataFileReadWriteObject(void)=0;
            
    };
}

#endif