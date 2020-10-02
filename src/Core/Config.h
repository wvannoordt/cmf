#ifndef CONFIG_H_CMF
#define CONFIG_H_CMF

#ifndef CMF_DIM
#define CMF_DIM 2
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

#endif
