NUM_PROC = 10

example: example.c
	mpicc example.c -o example

clean:
	touch example && rm example

run: example
	mpiexec -n $(NUM_PROC) example
