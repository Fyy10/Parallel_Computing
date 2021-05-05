SOURCE = nbody1
DEFINE = DEV,UNIFIED

main: $(SOURCE).cu
	nvcc -o nbody $(SOURCE).cu -D $(DEFINE)

run: main
	./nbody

prof: main
	nvprof ./nbody

clean:
	touch nbody && rm nbody
