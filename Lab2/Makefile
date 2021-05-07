SOURCE = nbody2
TARGET = nbody
# nBodies = 2 << ORDER_N
ORDER_N = 11
DEFINE = DEV
METRICS = achieved_occupancy,gld_efficiency,gst_efficiency,sm_efficiency,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_pipe_busy,shared_efficiency

main: $(SOURCE).cu
	nvcc -o $(TARGET) $(SOURCE).cu -D $(DEFINE)

run: main
	./$(TARGET)

prof: main
	nvprof -m $(METRICS) ./$(TARGET)

benchmark: main benchmark.py
	python benchmark.py $(TARGET) $(ORDER_N)

test: main test.py
	python test.py $(TARGET)

clean:
	touch $(TARGET) && rm $(TARGET)
