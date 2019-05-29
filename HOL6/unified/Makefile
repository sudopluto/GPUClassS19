all:
	g++ -O3 main.cpp -o cpu
	nvcc -arch=sm_35 -lineinfo -O3 main.cu -o gpu

clean:
	rm cpu gpu *.out
