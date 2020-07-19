MAINCC := $(wildcard *.cc)
TARGET := test_$(basename $(notdir ${MAINCC}))
main:
	${CC_HOST} ${CURRENT_ICONFIG} ${MAINCC} -o ${TARGET} ${CURRENT_LCONFIG}

clean:
	-rm ${TARGET}

test: clean main
	./${TARGET}
