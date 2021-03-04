#include "pch.h"
#include <iostream>
#include <fstream>

#include <time.h>
#include <string>

struct base
{
	float real;
	float imaginary;

	float magnitude()
	{
		return sqrtf(real * real + imaginary * imaginary);
	}

	friend base operator * (const base& a, const base& b);
	friend base operator + (const base& a, const base& b);
	friend base operator - (const base& a, const base& b);

	friend base operator / (const base& left, const float& right);
};
base operator * (const base& a, const base& b)
{
	return { a.real * b.real - a.imaginary * b.imaginary, a.imaginary * b.real + a.real * b.imaginary };
}
base operator + (const base& a, const base& b)
{
	return { a.real + b.real, a.imaginary + b.imaginary };
}
base operator - (const base& a, const base& b)
{
	return { a.real - b.real, a.imaginary - b.imaginary };
}
base operator / (const base& a, const float& b)
{
	return { a.real / b, a.imaginary / b };
}

#define PI 3.141592654f


enum resultFFT { success, sizeError };

resultFFT FFTonCPU(base *a, size_t size)
{
	// Size check for a power of two.
	int l = 1;
	while (l < size)
		l = l << 1;
	if (l != size)
		return sizeError;


	
	// Relocate input vector for FFT.
	for (int i = 1, j = 0; i < size; i++)
	{
		int bit = size >> 1;
		for (; j >= bit; bit >>= 1)
			j -= bit;
		j += bit;
		if (i < j)
			std::swap(a[i], a[j]);
	}

	// log2_N iteration of FFT.
	for (unsigned int len = 2; len <= size; len <<= 1)
	{
		float angle = 2 * PI / len * (-1);
		base wlen = { cosf(angle), sinf(angle) };
		for (int i = 0; i < size; i += len)
		{
			base w = { 1, 0 };
			for (int j = 0; j < len / 2; j++)
			{
				base u = a[i + j], v = a[i + j + len / 2] * w;
				a[i + j] = u + v;
				a[i + j + len / 2] = u - v;
				w = w * wlen;
			}
		}
	}

	// Normalization.
	for (int i = 0; i < size; i++)
		a[i] = a[i] / size;

	return success;
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

void WriteToFile(std::string fileName, bool success, std::string message, base *a, size_t size, float t)
{
	std::ofstream file;
	file.open(fileName);

	file << message << std::endl;

	if (success)
	{
		file << "Executing time: " << t << " seconds" << std::endl;
		file << "Size of input vector: " << size << std::endl;
		file << "Result of FFT:" << std::endl;
		file << "(real) (imaginary)" << std::endl;
		for (int i = 0; i < size; i++)
			file << a[i].real << " " << a[i].imaginary << std::endl;
	}

	file.close();
}

void OneLaunchFFT(std::string fileNameInput, std::string fileNameOutput, std::ostream& profStream)
{
	std::cout << fileNameInput << " is being read..." << std::endl;

	// Reading input vector from file.
	size_t size;
	float *a = ReadFromFile(size, fileNameInput);

	std::cout << "FFT for " << fileNameInput << " is being run..." << std::endl;

	// Initialization complex vector.
	base *b = new base[size];
	for (int i = 0; i < size; i++)
		b[i] = { a[i], 0 };


	float exTime;
	exTime = clock();

	// Executing FFT.
	resultFFT result = FFTonCPU(b, size);

	exTime = (float)(clock() - exTime) / CLOCKS_PER_SEC;


	std::cout << "FFT for " << fileNameInput << " is done" << std::endl;

	std::string message = "";

	if (result == success)
	{
		message = "Success!";
		profStream << "total executing time: " << exTime << " s." << std::endl;

		//std::cout << "Result of FFT for " << fileNameInput << " is being read in " << fileNameOutput << std::endl;
		//WriteToFile(fileNameOutput, result == success, message, b, size, exTime);
	}
	else if (result == sizeError)
		message = "Size error!";
	else
		message = "Cuda error!";
}



int main()
{
	std::ofstream profFile_512;
	profFile_512.open("CPU_prof_512.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_512 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_512.txt", "GPU_output_512.txt", profFile_512);
		profFile_512 << std::endl << std::endl << std::endl;
	}

	profFile_512.close();



	std::ofstream profFile_8192;
	profFile_8192.open("CPU_prof_8192.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_8192 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_8192.txt", "GPU_output_8192.txt", profFile_8192);
		profFile_8192 << std::endl << std::endl << std::endl;
	}

	profFile_8192.close();



	std::ofstream profFile_131072;
	profFile_131072.open("CPU_prof_131072.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_131072 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_131072.txt", "GPU_output_131072.txt", profFile_131072);
		profFile_131072 << std::endl << std::endl << std::endl;
	}

	profFile_131072.close();



	std::ofstream profFile_2097152;
	profFile_2097152.open("CPU_prof_2097152.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_2097152 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_2097152.txt", "GPU_output_2097152.txt", profFile_2097152);
		profFile_2097152 << std::endl << std::endl << std::endl;
	}

	profFile_2097152.close();



	std::ofstream profFile_33554432;
	profFile_33554432.open("CPU_prof_33554432.txt");

	for (int i = 0; i < 20; i++)
	{
		//profFile_33554432 << "For file input_33554432.txt " << "(" << i << "):" << std::endl;
		OneLaunchFFT("input_33554432.txt", "GPU_output_33554432.txt", profFile_33554432);
		profFile_33554432 << std::endl << std::endl << std::endl;
	}

	profFile_33554432.close();
	
	return 0;
}