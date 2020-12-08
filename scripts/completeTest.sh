trap "exit" INT
CMF_MAIN_DIR=..
cd ${CMF_MAIN_DIR}
start=`date +%s`
make clean && \
DIM=2 OPTLEVEL=0 MEMCHECK=1 PARALLEL=0 make test &&\
make clean &&\
DIM=2 OPTLEVEL=3 MEMCHECK=1 PARALLEL=0 make test &&\
make clean &&\
DIM=3 OPTLEVEL=0 MEMCHECK=1 PARALLEL=0 make test &&\
make clean &&\
DIM=3 OPTLEVEL=3 MEMCHECK=1 PARALLEL=0 make test &&\
make clean &&\
DIM=2 OPTLEVEL=3 MEMCHECK=0 PARALLEL=1 make test &&\
make clean &&\
DIM=3 OPTLEVEL=3 MEMCHECK=0 PARALLEL=1 make test &&\
end=`date +%s` &&\
runtime=$((end-start)) &&\
cd scripts &&\
echo "Total runtime, date: ${runtime}, ${end}" >> .cmf.testdurations.cmflog &&\
echo "[logged]"
    
