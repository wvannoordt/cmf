#ifndef ICMF_DATABASE_READ_WRITE_OBJECT_H
#define ICMF_DATABASE_READ_WRITE_OBJECT_H

namespace cmf
{
    /// @brief Represents an object that can read from and written to a CMF data file
    /// @author WVN
    class ICmfDataBaseReadWriteObject
    {
        public:
            
            /// @brief Constructor
            /// @author WVN
            ICmfDataBaseReadWriteObject(void);
            
            /// @brief Destructor
            /// @author WVN
            virtual ~ICmfDataBaseReadWriteObject(void){};
            
    };
}

#endif