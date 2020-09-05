#ifndef CONFIG_H_ANAPTRIC
#define CONFIG_H_ANAPTRIC

#ifndef DIM
#define DIM 2
#endif

#if(DIM==2)
#define IS3D 0
#define __only2d(mycode) {mycode;}
#define __only3d(mycode) ;
#else
#define IS3D 1
#define __only2d(mycode) ;
#define __only3d(mycode) {mycode;}
#endif

#ifndef PTL_AUTO_ENUM_TERMINATOR
#define PTL_AUTO_ENUM_TERMINATOR ""
#endif

#endif
