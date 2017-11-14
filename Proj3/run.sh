#!/bin/sh
module purge
module load mvapich2-1.9a2/gnu-4.6.2
module load gcc-4.6.2

echo "Yue Wang's script starts, submitting jobs..."
if [ ! -f $PWD/results11 ]
then 
    make test1
fi

if [ ! -f $PWD/results21 ]
then 
    make test2
fi

if [ ! -f $PWD/results31 ]
then 
    make test3
fi

echo "Job submitted, waiting for completion..."
for (( i = 1; i <= 3; i++))
do
    for (( j = 1; j <= 4; j++))
    do
        while [ ! -f $PWD/results$i$j ]
        do
            sleep 2
        done
    done
done

echo "Jobs done, preparing final result..."
rm results
for (( i = 1; i <= 3; i++))
do
    echo Part$i >> results
    for (( j = 1; j <= 4; j++))
    do
        cat $PWD/results$i$j >> results
    done
    echo "" >> results
done

echo "Results generated."








