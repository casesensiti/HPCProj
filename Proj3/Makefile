test1:   part1
	qsub -v prog=./part1,JOB_PATH=`pwd` -o results11 jobfile_1
	qsub -v prog=./part1,JOB_PATH=`pwd` -o results12 jobfile_2
	qsub -v prog=./part1,JOB_PATH=`pwd` -o results13 jobfile_3
	qsub -v prog=./part1,JOB_PATH=`pwd` -o results14 jobfile_4
test2:   part2
	qsub -v prog=./part2,JOB_PATH=`pwd` -o results21 jobfile_1
	qsub -v prog=./part2,JOB_PATH=`pwd` -o results22 jobfile_2
	qsub -v prog=./part2,JOB_PATH=`pwd` -o results23 jobfile_3
	qsub -v prog=./part2,JOB_PATH=`pwd` -o results24 jobfile_4
test3:   part3
	qsub -v prog=./part3,JOB_PATH=`pwd` -o results31 jobfile_1
	qsub -v prog=./part3,JOB_PATH=`pwd` -o results32 jobfile_2
	qsub -v prog=./part3,JOB_PATH=`pwd` -o results33 jobfile_3
	qsub -v prog=./part3,JOB_PATH=`pwd` -o results34 jobfile_4
local: part1 part2
	./part1 500
	./part2 500 20
part1: part1.c
	mpicc -std=c99 -o part1 part1.c	
part2: part2.c
	mpicc -std=c99 -o part2 part2.c	
part3: part3.c
	mpicc -std=c99 -o part3 part3.c	
clean:
	rm part1 part2 part3
