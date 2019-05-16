.PHONY: all clean exec exec_shared

all: stencil stencil_shared

stencil: stencil.cu
	nvcc -O3 -arch=sm_35 -lineinfo stencil.cu -o stencil

stencil_shared: stencil_shared.cu
	nvcc -O3 -arch=sm_35 -lineinfo stencil_shared.cu -o stencil_shared

clean:
	rm stencil stencil_shared *.out

exec:
	sbatch exec.bash

exec_shared:
	sbatch exec_shared.bash
