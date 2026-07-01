FUN_LIBS_BUILD=../proxy-FUn/FUn-Libs/build-cuda source env.sh && python3 workflows/modeling/examples/adt/acoustic_modeling.py \
    --decision workflows/modeling/examples/inputfiles/acquisition/decision \
    --database identity \
    --lx 1500 --ly 1500 --lz 1500 \
    --ex 100 --ey 100 --ez 100 \
    --vp 1500 --dt 0.001 --tmax 1.5 --f0 5 \
    --output-format segy \
    --output-dir seismoTraces 2>&1 | head -80
