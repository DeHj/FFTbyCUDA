#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
	 
#include <algorithm>
#include <math.h>
#include <cuComplex.h>

#include <time.h>
#include <string>


#define PI 3.141592654f

__global__ void kernel_FFT(cuFloatComplex *in, cuFloatComplex *out, unsigned int halfLength, unsigned int level)
{
	int linearIndex = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * (blockDim.x * blockDim.y);

	// Find accordance between thread index and processing item index.
	unsigned int fftIndex = (linearIndex >> level) << (level + 1);
	fftIndex = fftIndex | (linearIndex & ((1 << level) - 1));

	cuFloatComplex v = in[fftIndex + halfLength];
	float angle = (-1) * PI / halfLength * (fftIndex & ((1 << level) - 1));
	v = cuCmulf(v, make_cuFloatComplex(cosf(angle), sinf(angle)));

	// Butterfly step.
	out[fftIndex] = cuCaddf(in[fftIndex], v);
	out[fftIndex + halfLength] = cuCsubf(in[fftIndex], v);
}
__global__ void kernel_Copy(cuFloatComplex *from, cuFloatComplex *to)
{
	int linearIndex = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * (blockDim.x * blockDim.y);
	to[linearIndex] = from[linearIndex];
}
__global__ void kernel_Normalization(cuFloatComplex *a, float koef)
{
	int linearIndex = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * (blockDim.x * blockDim.y);
	a[linearIndex] = make_cuFloatComplex(a[linearIndex].x / koef, a[linearIndex].y / koef);
}
__global__ void kernel_Relocation(cuFloatComplex *from, cuFloatComplex *to, unsigned int log2_N) // level = log2_N
{
	int linearIndex = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * (blockDim.x * blockDim.y);

	int oldIndex = linearIndex;
	int newIndex = 0;
	for (int i = 0; i < log2_N - 1; i++)
	{
		newIndex += oldIndex ^ ((oldIndex >> 1) << 1);
		newIndex <<= 1;
		oldIndex >>= 1;
	}
	newIndex += oldIndex ^ ((oldIndex >> 1) << 1);

	to[linearIndex] = from[newIndex];
}

__global__ void CUDA_printf(cuFloatComplex *a)
{
	int linearIndex = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * (blockDim.x * blockDim.y);

	printf("%f %f \n", a[linearIndex].x, a[linearIndex].y);
}



const dim3 threadsInBlock = dim3(16, 16);


struct measurement
{
	const char *valueName;
	float time;
};

enum resultFFT { success, sizeError, cudaError };

resultFFT FFTwithCuda(cuFloatComplex *a, size_t size, measurement **measurements, int& n_meas)
{
	float time_transferOnDevice;
	float time_relocating;
	float time_FFTCycle;
	float time_normalization;
	float time_transferFromDevice;
	float time_dataFree;


	// Size check for a power of two.
	int l = 1;
	int log2_N = 0;
	while (l < size)
	{
		log2_N++;
		l = l << 1;
	}
	if ((l != size) || (size < threadsInBlock.x * threadsInBlock.y * threadsInBlock.z * 2))
		return sizeError;

	// Choose which GPU to run on, change this on a multi-	GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) goto CudaError;

	

	time_transferOnDevice = clock();

	// Allocate GPU buffers and copy input vector from host memory to GPU buffers.
	cuFloatComplex* dev_in;
	cuFloatComplex* dev_out;
	cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(cuFloatComplex));
	if (cudaStatus != cudaSuccess) goto CudaError;
	cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(cuFloatComplex));
	if (cudaStatus != cudaSuccess) goto CudaError;
	cudaStatus = cudaMemcpy(dev_out, a, size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) goto CudaError;

	time_transferOnDevice = (float)(clock() - time_transferOnDevice) / CLOCKS_PER_SEC;



	int nBlocks = size / (threadsInBlock.x * threadsInBlock.y);



	time_relocating = clock();

	// Relocate input vector for FFT.
	kernel_Relocation <<< nBlocks, threadsInBlock >>> (dev_out, dev_in, log2_N);

	time_relocating = (float)(clock() - time_relocating) / CLOCKS_PER_SEC;



	time_FFTCycle = clock();

	// log2_N iteration of FFT.
	for (unsigned int len = 2, level = 0; len <= size; len <<= 1, level++)
	{
		// Launch a kernel on the GPU with one thread for every two element.
		kernel_FFT <<< nBlocks / 2, threadsInBlock >>> (dev_in, dev_out, len / 2, level);
		
		// Synchronize.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) goto CudaError;

		// Copy from dev_to to dev_from.
		kernel_Copy <<< nBlocks, threadsInBlock >>> (dev_out, dev_in);

		// Synchronize.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) goto CudaError;
	}

	time_FFTCycle = (float)(clock() - time_FFTCycle) / CLOCKS_PER_SEC;



	time_normalization = clock();

	// Normalization.
	kernel_Normalization <<< nBlocks, threadsInBlock >>> (dev_in, size);

	time_normalization = (float)(clock() - time_normalization) / CLOCKS_PER_SEC;



	time_transferFromDevice = clock();

	// Copy result from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(a, dev_in, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto CudaError;

	time_transferFromDevice = (float)(clock() - time_transferFromDevice) / CLOCKS_PER_SEC;



	time_dataFree = clock();

	// Clear GPU memory.
	cudaFree(dev_in);
	cudaFree(dev_out);
	
	cudaDeviceReset();

	time_dataFree = (float)(clock() - time_dataFree) / CLOCKS_PER_SEC;


	
	*measurements = new measurement[6];
	(*measurements)[0] = { "time of data allocating and transfer from host on device", time_transferOnDevice };
	(*measurements)[1] = { "relocating time", time_relocating };
	(*measurements)[2] = { "FFT cycle time", time_FFTCycle };
	(*measurements)[3] = { "normalization time", time_normalization };
	(*measurements)[4] = { "time of data transfer from device on host", time_transferFromDevice };
	(*measurements)[5] = { "time of data free on device", time_dataFree };
	n_meas = 6;

	return success;

CudaError:
	// Clear GPU memory.
	cudaFree(dev_in);
	cudaFree(dev_out);

	cudaDeviceReset();

	return cudaError;
}



float* ReadFromFile(size_t &size, std::string fileName)
{
	std::ifstream file;
	file.open(fileName);

	file >> size;

	float *result = new float[size];

	for (int i = 0; i < size; i++)
		file >> result[i];

	file.close();

	return result;
}

void Prof(std::ostream& stream,
	bool success, std::string message,
	float totalTime, measurement *measurements, int& n_meas)
{
	stream << message << std::endl;

	if (success)
	{
		stream << "total executing time: " << totalTime << " s." << std::endl;
		for (int i = 0; i < n_meas; i++)
			stream << measurements[i].valueName << ": " << measurements[i].time << " s." << std::endl;
	}
}

void WriteToFile(std::string fileName, cuFloatComplex *a, size_t size)
{
	std::ofstream file;
	file.open(fileName);

	file << "Size is " << size << std::endl;
	file << "(real) (imaginary)" << std::endl;
	for (int i = 0; i < size; i++)
		file << a[i].x << " " << a[i].y << std::endl;

}

void OneLaunchFFT(std::string fileNameInput, std::string fileNameOutput, std::ostream& profStream)
{
	std::cout << fileNameInput << " is being read..." << std::endl;

	// Reading input vector from file.
	size_t size;
	float *a = ReadFromFile(size, fileNameInput);



	std::cout << "FFT for " << fileNameInput << " is being run..." << std::endl;

	// Initialization complex vector.
	cuFloatComplex *b = new cuFloatComplex[size];
	for (int i = 0; i < size; i++)
		b[i] = make_cuFloatComplex(a[i], 0);



	int n_meas;
	measurement *measurements = new measurement;
	
	float exTime;
	exTime = clock();

	// Executing FFT.
	resultFFT result = FFTwithCuda(b, size, &measurements, n_meas);

	exTime = (float)(clock() - exTime) / CLOCKS_PER_SEC;
	


	std::cout << "FFT for " << fileNameInput << " is done" << std::endl;


	std::string message = "";

	if (result == resultFFT::success)
	{
		message = "";
		std::cout << "Success!" << std::endl;
		//std::cout << "Result of FFT for " << fileNameInput << " is being write in " << fileNameOutput << std::endl;
		//WriteToFile(fileNameOutput, b, size);
	}
	else if (result == resultFFT::sizeError)
	{
		message = "Size error!";
		std::cout << "Size error!" << std::endl;
	}
	else
	{
		message = "Cuda error!";
		std::cout << "Cuda error!" << std::endl;
	}
	
	Prof(profStream, result == resultFFT::success, message, exTime, measurements, n_meas);
}

int main()
{
	std::ofstream profFile_512;
	profFile_512.open("GPU_prof_512.txt");
	
	for (int i = 0; i < 20; i++)
	{
		//profFile_512 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_512.txt", "GPU_output_512.txt", profFile_512);
		profFile_512 << std::endl << std::endl << std::endl;
	}

	profFile_512.close();

	

	std::ofstream profFile_8192;
	profFile_8192.open("GPU_prof_8192.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_8192 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_8192.txt", "GPU_output_8192.txt", profFile_8192);
		profFile_8192 << std::endl << std::endl << std::endl;
	}

	profFile_8192.close();



	std::ofstream profFile_131072;
	profFile_131072.open("GPU_prof_131072.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_131072 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_131072.txt", "GPU_output_131072.txt", profFile_131072);
		profFile_131072 << std::endl << std::endl << std::endl;
	}

	profFile_131072.close();



	std::ofstream profFile_2097152;
	profFile_2097152.open("GPU_prof_2097152.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_2097152 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_2097152.txt", "GPU_output_2097152.txt", profFile_2097152);
		profFile_2097152 << std::endl << std::endl << std::endl;
	}

	profFile_2097152.close();



	std::ofstream profFile_33554432;
	profFile_33554432.open("GPU_prof_33554432.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_33554432 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_33554432.txt", "GPU_output_33554432.txt", profFile_33554432);
		profFile_33554432 << std::endl << std::endl << std::endl;
	}

	profFile_33554432.close();

	return 0;
}