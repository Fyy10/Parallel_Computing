SOURCE = nbody2
DEFINE = DEV
METRICS = achieved_occupancy,gld_efficiency,gst_efficiency,sm_efficiency,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_pipe_busy,shared_efficiency

main: $(SOURCE).cu
	nvcc -o nbody $(SOURCE).cu -D $(DEFINE)

run: main
	./nbody

prof: main
	nvprof -m $(METRICS) ./nbody

clean:
	touch nbody && rm nbody
