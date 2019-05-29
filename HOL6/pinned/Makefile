.PHONY: all clean

all: compile

compile:
	nvcc -arch=sm_35 -lineinfo -O3 vadd.cu -o vadd
	nvcc -arch=sm_35 -lineinfo -O3 vadd_pinned.cu -o vadd_pinned

clean:
	rm vadd 

clean-log:
	rm *.out
