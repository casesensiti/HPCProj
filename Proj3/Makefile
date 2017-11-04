tardis: part1 part2
	qsub -v msize=1000,JOB_PATH=`pwd` -o job1_1000.out jobfile_1
	qsub -v msize=2000,JOB_PATH=`pwd` -o job1_2000.out jobfile_1
	qsub -v msize=3000,JOB_PATH=`pwd` -o job1_3000.out jobfile_1
	qsub -v msize=4000,JOB_PATH=`pwd` -o job1_4000.out jobfile_1
	qsub -v msize=5000,JOB_PATH=`pwd` -o job1_5000.out jobfile_1
	qsub -v msize=1000,JOB_PATH=`pwd`,step=20 -o job2_1000_20.out jobfile_2
	qsub -v msize=2000,JOB_PATH=`pwd`,step=20 -o job2_2000_20.out jobfile_2

local: part1 part2
	./part1 500
	./part2 500 20
part1: part1.cpp
	g++ -o part1 part1.cpp -I/opt/lapack/include /opt/lapack/lib/liblapacke.a /opt/lapack/lib/liblapack.a /opt/lapack/lib/librefblas.a -lgfortran -lm
	
part2: part2.cpp
	g++ -O3 -o part2 part2.cpp -I/opt/lapack/include /opt/lapack/lib/liblapacke.a /opt/lapack/lib/liblapack.a /opt/lapack/lib/librefblas.a -lgfortran -lm

clean:
	rm part1 part2
