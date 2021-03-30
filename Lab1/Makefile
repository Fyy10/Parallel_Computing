TARGET_NAME = Lab1
SOURCE_NAME = Lab1
NUM_PROC = 6
LIBS = -lm
PRIME_N = 1000

main: $(SOURCE_NAME).c
	mpicc -o $(TARGET_NAME) $(SOURCE_NAME).c $(LIBS)

debug: $(SOURCE_NAME).c
	mpicc -g -o $(TARGET_NAME) $(SOURCE_NAME).c $(LIBS)

clean:
	touch $(TARGET_NAME) && rm $(TARGET_NAME)

run: main
	mpiexec -n $(NUM_PROC) $(TARGET_NAME) $(PRIME_N)
