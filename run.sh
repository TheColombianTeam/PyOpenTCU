#!/usr/bin/env bash
DIR=`pwd`
DIR_RESULTS=${DIR}/faults
FAULTS_FILE=${DIR_RESULTS}/fault.csv
FAULTS=$(cat $FAULTS_FILE)

for matrix in 0 1 2 3 4 5 6 7 8 9
do
    echo "target_fault,target_thread_group_fault,position_fault,mask_fault,type_fault,position,golden_data,mask_golden,faulty_data,mask_faulty,mask,error_relative,error_abs" > ${DIR_RESULTS}/results_$matrix.csv
    IDX=0
    python sim.py None run > ${DIR_RESULTS}/golden_$matrix.txt
    cp ${DIR_RESULTS}/a.npy ${DIR_RESULTS}/a_$matrix.npy
    cp ${DIR_RESULTS}/b.npy ${DIR_RESULTS}/b_$matrix.npy
    cp ${DIR_RESULTS}/c.npy ${DIR_RESULTS}/c_$matrix.npy
    for fault in $FAULTS
    do
        echo "Matrix: $matrix, Fault: $IDX"
        python sim.py $IDX run > ${DIR_RESULTS}/result.txt
        DIFF=$(diff --suppress-common-lines -y ${DIR_RESULTS}/golden_$matrix.txt ${DIR_RESULTS}/result.txt) 
        if [ "$DIFF" ]
        then
            diff --suppress-common-lines -y ${DIR_RESULTS}/golden_$matrix.txt ${DIR_RESULTS}/result.txt > ${DIR_RESULTS}/diff.txt
            python sim.py $IDX results >> ${DIR_RESULTS}/results_$matrix.csv
        fi
        IDX=$((IDX+1))
    done
done
