#!/usr/bin/env bash
DIR=`pwd`
DIR_RESULTS=${DIR}/faults
FAULTS_FILE=${DIR_RESULTS}/fault.csv
FAULTS=$(cat $FAULTS_FILE)
IDX=0

python sim.py None run > ${DIR_RESULTS}/golden.txt
for fault in $FAULTS
do
    echo $IDX
    python sim.py $IDX run > ${DIR_RESULTS}/result.txt
    DIFF=$(diff --suppress-common-lines -y ${DIR_RESULTS}/golden.txt ${DIR_RESULTS}/result.txt) 
    if [ "$DIFF" ]
    then
        diff --suppress-common-lines -y ${DIR_RESULTS}/golden.txt ${DIR_RESULTS}/result.txt > ${DIR_RESULTS}/diff.txt
        python sim.py $IDX results >> ${DIR_RESULTS}/results.csv
    fi
    IDX=$((IDX+1))
done
