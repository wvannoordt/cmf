#ifndef CONFIG_H_GTREE
#define CONFIG_H_GTREE

#ifndef DIM
#define DIM 2
#endif

#if(DIM==2)
#define IS3D 0
#else
#define IS3D 1
#endif

#ifndef PTL_AUTO_ENUM_TERMINATOR
#define PTL_AUTO_ENUM_TERMINATOR ""
#endif

#endif
