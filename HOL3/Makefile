vadd: vadd.cu
	nvcc -arch=sm_35 -O3 vadd.cu -o vadd

vadd_coalesced: vadd_coalesced.cu
	nvcc -arch=sm_35 -O3 vadd_coalesced.cu -o vadd_coalesced

.PHONY: clean

clean :
	rm vadd vadd_coalesced
