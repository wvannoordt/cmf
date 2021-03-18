LIB_NAME := cmf

ifndef OPTLEVEL
OPTLEVEL := 0
endif

ifndef CUDA_ENABLE
CUDA_ENABLE := 0
endif

ifndef DEBUG_LEVEL
DEBUG_LEVEL := 0
endif

ifndef MEMCHECK
MEMCHECK := 0
endif

ifndef DIM
DIM := 2
endif

ifndef DOCLEAN
DOCLEAN :=
endif

ifndef ZLIB_ENABLE
ZLIB_ENABLE := 1
endif

ifndef PARALLEL
PARALLEL := 1
endif
export PARALLEL

ifndef DOLATEXOUTPUT
DOLATEXOUTPUT := 0
endif
export DOLATEXOUTPUT

ifndef GPROF_ENABLE
GPROF_ENABLE := 0
endif

ifndef CUDA_ENABLE
CUDA_ENABLE := 1
endif

ifeq (${GPROF_ENABLE}, 1)
GPROF_FLAG := -pg
else
GPROF_FLAG :=
endif

CURRENT_BASEIDIR   = $(shell pwd)
CURRENT_SRC_DIR   := ${CURRENT_BASEIDIR}/src
CURRENT_LIB_DIR   := ${CURRENT_BASEIDIR}/lib
CURRENT_OBJ_DIR   := ${CURRENT_BASEIDIR}/obj
CURRENT_HDR_DIR   := ${CURRENT_BASEIDIR}/include
CURRENT_DOC_DIR   := ${CURRENT_BASEIDIR}/docs
CURRENT_TST_DIR   := ${CURRENT_BASEIDIR}/testing
CURRENT_HDRHX_DIR := ${CURRENT_BASEIDIR}/includex

#this path will need to be changed.
CURRENT_SUB_DIR   := ${CURRENT_BASEIDIR}/..
CURRENT_PTL_DIR   := ${CURRENT_SUB_DIR}/PTL


IFLAGS_DEPENDENCIES :=
IFLAGS_DEPENDENCIES += -I${CURRENT_PTL_DIR}/include

LFLAGS_DEPENDENCIES :=
LFLAGS_DEPENDENCIES += -L${CURRENT_PTL_DIR}/lib -lPTL

ifeq (${ZLIB_ENABLE}, 1)
LFLAGS_DEPENDENCIES += -lz
endif

CURRENT_IFLAGS := -I${CURRENT_HDR_DIR} -I${CURRENT_HDRHX_DIR}

SRC_FILES_HYBRID_H      := $(shell find ${CURRENT_SRC_DIR} -name *.cppx)
SRC_FILES_HYBRID_D      := $(shell find ${CURRENT_SRC_DIR} -name *.cppx)
SRC_FILES_HOST          := $(shell find ${CURRENT_SRC_DIR} -name *.cpp)
SRC_FILES_CUDA          := $(shell find ${CURRENT_SRC_DIR} -name *.cu)

HEADER_FILES    := $(shell find ${CURRENT_SRC_DIR} -name *.h)
HEADER_FILES_HX := $(shell find ${CURRENT_SRC_DIR} -name *.hx)

ifndef TESTS
TESTS := $(wildcard ${CURRENT_TST_DIR}/*)
else
TESTS := $(addprefix ${CURRENT_TST_DIR}/, ${TESTS})
endif


TARGET := ${CURRENT_LIB_DIR}/lib${LIB_NAME}.a

PY_EXE := $(shell which python3)
CC_HOST := $(shell which mpicxx)
CC_DEVICE :=  $(shell which nvcc) -ccbin=${CC_HOST}
ifneq (${PARALLEL}, 1)
CC_HOST := g++
CC_DEVICE := none
endif

CU_O_TARGET_NAME := ${CURRENT_OBJ_DIR}/CU_dlink.o
ifeq (${CUDA_ENABLE}, 1)
LINK_STEP := link_step
CU_O_TARGET := ${CU_O_TARGET_NAME}
ICUDA := -I/usr/local/cuda/include
LCUDA := -L/usr/local/cuda/lib64 -lcudadevrt -lcudart
else
LINK_STEP :=
OBJ_FILES_HYBRID_DEVICE :=
CUDA_ENABLE := 0
OBJ_FILES_CUDA :=
CU_O_TARGET :=
ICUDA :=
LCUDA :=
endif

VALGRIND_SUPPRESSION_FILE := ${CURRENT_BASEIDIR}/util/cmf_valgrind.supp
ifeq (${MEMCHECK}, 1)
VALG := $(shell which valgrind) --error-exitcode=166 --leak-check=full --error-limit=no --suppressions=${VALGRIND_SUPPRESSION_FILE}
else
VALG :=
endif
export VALG

COMPILE_TIME_OPT :=
COMPILE_TIME_OPT += -DCMF_DIM=${DIM}
COMPILE_TIME_OPT += -DCMF_ZLIB_EXT_ENABLE=${ZLIB_ENABLE}
COMPILE_TIME_OPT += -DCMF_ENABLE_STACK_BLOB=0
COMPILE_TIME_OPT += -DCMF_PARALLEL=${PARALLEL}
COMPILE_TIME_OPT += -DCUDA_ENABLE=${CUDA_ENABLE}
COMPILE_TIME_OPT += -DGLIBCXX_FORCE_NEW=1



DEVICE_FLAGS := -O${OPTLEVEL} -x cu -rdc=true -Xcompiler -fPIC ${COMPILE_TIME_OPT} -dc
DEVICE_DLINK_FLAGS := -Xcompiler -fPIC -rdc=true -dlink
HOST_FLAGS := -O${OPTLEVEL} -x c++ -g -Wno-unknown-pragmas -fPIC -fpermissive -std=c++11 -Werror -c ${LCUDA} ${GPROF_FLAG}

export CUDA_ENABLE

LZLIB :=
ifeq (${ALLOW_DEBUG_EXT}, 1)
LZLIB := -lz
endif


DO_CLEAN :=
ifeq (1, ${CUDA_ENABLE})
ifeq (,$(wildcard ${CU_O_TARGET_NAME}))
#DO_CLEAN := clean
endif
endif

ifeq (0, ${CUDA_ENABLE})
ifneq (,$(wildcard ${CU_O_TARGET_NAME}))
#DO_CLEAN := clean
endif
endif

ifeq (0, ${CUDA_ENABLE})
SRC_FILES_HYBRID_D :=
endif

CURRENT_IFLAGS += ${ICUDA}


TARGETNAME_HYBRID_H := $(addprefix ${CURRENT_OBJ_DIR}/,$(addsuffix .o,$(notdir ${SRC_FILES_HYBRID_H})))
TARGETNAME_HYBRID_D := $(addprefix ${CURRENT_OBJ_DIR}/,$(addsuffix .o,$(notdir ${SRC_FILES_HYBRID_D})))
TARGETNAME_HOST     := $(addprefix ${CURRENT_OBJ_DIR}/,$(addsuffix .o,$(notdir ${SRC_FILES_HOST})))
TARGETNAME_CUDA     := $(addprefix ${CURRENT_OBJ_DIR}/,$(addsuffix .o,$(notdir ${SRC_FILES_CUDA})))

SRC_FILES_HOST_DIR := $(dir ${SRC_FILES_HOST})
MP:=%

ifeq (,${TARGETNAME_HYBRID_D})
ifeq (,${TARGETNAME_CUDA})
LINK_STEP :=
endif
endif

export CURRENT_ICONFIG=-I${CURRENT_HDR_DIR} ${ICUDA} ${IFLAGS_DEPENDENCIES}
export CURRENT_LCONFIG= ${LCUDA} -L${CURRENT_LIB_DIR} -l${LIB_NAME} ${LFLAGS_DEPENDENCIES}
export CC_HOST
export CURRENT_BASEIDIR
export DIM

.PHONY: final docs

final: ${DO_CLEAN} PTL setup ${TARGETNAME_CUDA} ${TARGETNAME_HYBRID_D} ${LINK_STEP} ${TARGETNAME_HYBRID_H} ${TARGETNAME_HOST}
	${CC_HOST} -fPIC -shared ${GPROF_FLAG} ${CURRENT_OBJ_DIR}/*.o ${CURRENT_IFLAGS} ${IFLAGS_DEPENDENCIES} ${COMPILE_TIME_OPT} ${LZLIB} ${LCUDA} ${LFLAGS_DEPENDENCIES} -o ${TARGET}

.SECONDEXPANSION:
${TARGETNAME_HYBRID_D}: ${CURRENT_OBJ_DIR}/%.o : $$(filter $$(MP)/$$*,$$(SRC_FILES_HYBRID_D))
	${CC_DEVICE} ${DEVICE_FLAGS} ${COMPILE_TIME_OPT} ${CURRENT_IFLAGS} ${IFLAGS_DEPENDENCIES} $< -o $@

.SECONDEXPANSION:
${TARGETNAME_HYBRID_H}: ${CURRENT_OBJ_DIR}/%.o : $$(filter $$(MP)/$$*,$$(SRC_FILES_HYBRID_H))
	${CC_HOST} ${HOST_FLAGS} ${COMPILE_TIME_OPT} ${CURRENT_IFLAGS} ${IFLAGS_DEPENDENCIES} $< -o $@

.SECONDEXPANSION:
${TARGETNAME_HOST}: ${CURRENT_OBJ_DIR}/%.o : $$(filter $$(MP)/$$*,$$(SRC_FILES_HOST))
	${CC_HOST} ${HOST_FLAGS} ${COMPILE_TIME_OPT} ${CURRENT_IFLAGS} ${IFLAGS_DEPENDENCIES} $< -o $@

.SECONDEXPANSION:
${TARGETNAME_CUDA}: ${CURRENT_OBJ_DIR}/%.o : $$(filter $$(MP)/$$*,$$(SRC_FILES_CUDA))
	${CC_DEVICE} ${DEVICE_FLAGS} ${COMPILE_TIME_OPT} ${CURRENT_IFLAGS} ${IFLAGS_DEPENDENCIES} $< -o $@

${LINK_STEP}:
	${CC_DEVICE} ${DEVICE_DLINK_FLAGS} ${COMPILE_TIME_OPT} ${OBJ_FILES_HYBRID_DEVICE} ${OBJ_FILES_CUDA} -o ${CU_O_TARGET} -lcudadevrt

PTL:
	${MAKE} -C ${CURRENT_PTL_DIR} -f makefile

setup:
	echo ${AA}
	-rm -r ${CURRENT_HDR_DIR}
	-rm -r ${CURRENT_HDRHX_DIR}
	mkdir -p ${CURRENT_LIB_DIR}
	mkdir -p ${CURRENT_OBJ_DIR}
	mkdir -p ${CURRENT_HDR_DIR}
	mkdir -p ${CURRENT_HDRHX_DIR}
	@for hdr in ${HEADER_FILES} ; do\
		echo "Linking $${hdr}:";\
		ln -s $${hdr} -t ${CURRENT_HDR_DIR};\
	done
	@for hdr in ${HEADER_FILES_HX} ; do\
		echo "Linking $${hdr}:";\
		ln -s $${hdr} -t ${CURRENT_HDRHX_DIR};\
	done
	@for tdir in ${TESTS} ; do\
		echo "Linking ${CURRENT_BASEIDIR}/util/makefile.testing to $${tdir}/makefile";\
		ln -sf ${CURRENT_BASEIDIR}/util/makefile.testing $${tdir}/makefile;\
	done
	@for tdir in ${TESTS} ; do\
		echo "Linking ${CURRENT_BASEIDIR}/util/cmftestutils.h to $${tdir}/makefile";\
		ln -sf ${CURRENT_BASEIDIR}/util/cmftestutils.h $${tdir}/cmftestutils.h;\
	done

clean:
	-rm -r ${CURRENT_LIB_DIR}
	-rm -r ${CURRENT_OBJ_DIR}
	-rm -r ${CURRENT_HDR_DIR}
	-rm -r ${CURRENT_HDRHX_DIR}
	@for tdir in ${TESTS} ; do\
		${MAKE} -C $${tdir} -f makefile clean;\
		echo "removing $${tdir}/output";\
		unlink $${tdir}/makefile||echo"";\
		unlink $${tdir}/cmftestutils.h||echo"";\
		rm -rf $${tdir}/output;\
	done
	${MAKE} -C ${CURRENT_DOC_DIR} -f makefile.docs clean

test: ${DOCLEAN} final
	@for tdir in ${TESTS} ; do\
		${MAKE} -C $${tdir} -f makefile test || exit 4;\
	done
	@echo "ALL TESTS SUCCESSFUL"

docs:
	${MAKE} -C ${CURRENT_DOC_DIR} -f makefile.docs
