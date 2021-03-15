#ifndef CONFIG_H_CMF
#define CONFIG_H_CMF

//Dimension of the simulation
#ifndef CMF_DIM
#define CMF_DIM 2
#endif

//Enables stack allocation for large(ish) arrays
#ifndef CMF_ENABLE_STACK_BLOB
#define CMF_ENABLE_STACK_BLOB 0
#endif

#if(CMF_ENABLE_STACK_BLOB)
#ifndef CMF_STACK_BLOB_SIZE
#define CMF_STACK_BLOB_SIZE 1000000
#endif
#else
#define CMF_STACK_BLOB_SIZE 1
#endif

#if(CMF_DIM==2)
#define CMF_IS3D 0
#define __only2d(mycode) {mycode;}
#define __only3d(mycode) ;
#else
#define CMF_IS3D 1
#define __only2d(mycode) ;
#define __only3d(mycode) {mycode;}
#endif

#ifndef PTL_AUTO_ENUM_TERMINATOR
#define PTL_AUTO_ENUM_TERMINATOR ""
#endif

#define likely(x)    __builtin_expect (!!(x), 1)
#define unlikely(x)  __builtin_expect (!!(x), 0)

#endif
