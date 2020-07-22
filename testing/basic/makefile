MAINCC := $(wildcard *.cc)
TARGET := test_$(basename $(notdir ${MAINCC}))
main: setup
	${CC_HOST} ${CURRENT_ICONFIG} ${MAINCC} -o ${TARGET} ${CURRENT_LCONFIG}

setup:
	mkdir -p output

clean:
	-rm ${TARGET}

test: clean main
	./${TARGET}
