NVCC=nvcc 

#OPENCV_INCLUDE_PATH="$(OPENCV_ROOT)/include/opencv4"

#OPENCV_LD_FLAGS = -L $(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o jacob.o
	g++ -o gjacob main.o jacobian_kernel.o $(CUDA_LD_FLAGS)

main.o:main.cpp jacobian_kernel.h utils.h constants.h 
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) main.cpp 

jacob.o: jacobian_kernel.cu jacobian_kernel.h  utils.h constants.h
	$(NVCC) -c jacobian_kernel.cu $(NVCC_OPTS)

clean:
	rm *.o jacobian
