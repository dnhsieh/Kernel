// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include <cmath>
#include "besselk.h"
#include "polybesselk.h"
#include "matvec.h"
#include "constants.h"

void setBesselkCoefficients()
{
	cudaMemcpyToSymbol(c_P01Vec, P01Vec, sizeof(double) * (P01Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q01Vec, Q01Vec, sizeof(double) * (Q01Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P02Vec, P02Vec, sizeof(double) * (P02Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q02Vec, Q02Vec, sizeof(double) * (Q02Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P03Vec, P03Vec, sizeof(double) * (P03Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q03Vec, Q03Vec, sizeof(double) * (Q03Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P11Vec, P11Vec, sizeof(double) * (P11Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q11Vec, Q11Vec, sizeof(double) * (Q11Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P12Vec, P12Vec, sizeof(double) * (P12Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q12Vec, Q12Vec, sizeof(double) * (Q12Deg + 1), 0, cudaMemcpyHostToDevice);	

	cudaMemcpyToSymbol(c_P13Vec, P13Vec, sizeof(double) * (P13Deg + 1), 0, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(c_Q13Vec, Q13Vec, sizeof(double) * (Q13Deg + 1), 0, cudaMemcpyHostToDevice);	

	return;
}

__global__ void d2qGaussian(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat, double *d_btmMat,
                            double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec, biVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);
		getVector(biVec, d_btmMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec, bjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);
			getVector(bjVec, d_btmMat, lmkjIdx, lmkNum);

			vector qijVec, bijVec;
			vectorSubtract(qijVec, qiVec, qjVec);
			vectorSubtract(bijVec, biVec, bjVec);

			double dijSqu = eucnormSqu(qijVec) / knlWidthSqu;

			double lreVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                * 2.0 * exp(-dijSqu) / knlWidthSqu;
			double  bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lreVal * (bqVal * (2.0 / knlWidthSqu) * qijVec.x - bijVec.x);
			dqKVec.y += lreVal * (bqVal * (2.0 / knlWidthSqu) * qijVec.y - bijVec.y);
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void d2qiGaussian(double *d_dqiiKMat, double *d_dqjiKMat, double *d_lmkiMat, double *d_lmkjMat,
                             double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                             double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqiiKVec = {0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0};

		vector qiVec, liVec, biVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
		getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec, bjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
			getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijSqu = eucnormSqu(qijVec) / knlWidthSqu;

			double lreVal = dotProduct(liVec, rjVec) * 2.0 * exp(-dijSqu) / knlWidthSqu;
			double biqVal =  dotProduct(biVec, qijVec);
			double bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lreVal * (biqVal * (2.0 / knlWidthSqu) * qijVec.x - biVec.x);
			dqiiKVec.y += lreVal * (biqVal * (2.0 / knlWidthSqu) * qijVec.y - biVec.y);

			dqjiKVec.x += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qijVec.x + bjVec.x);
			dqjiKVec.y += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qijVec.y + bjVec.y);
		}

		setVector(d_dqiiKMat, dqiiKVec, lmkiIdx, lmkiNum);
		setVector(d_dqjiKMat, dqjiKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void d2qjGaussian(double *d_dqijKMat, double *d_dqjjKMat, double *d_lmkiMat, double *d_lmkjMat,
                             double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                             double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqijKVec = {0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0};

		vector qjVec, rjVec, bjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
		getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec, biVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
			getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijSqu = eucnormSqu(qjiVec) / knlWidthSqu;

			double lreVal = dotProduct(liVec, rjVec) * 2.0 * exp(-dijSqu) / knlWidthSqu;
			double biqVal = -dotProduct(biVec, qjiVec);
			double bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lreVal * (biqVal * (2.0 / knlWidthSqu) * qjiVec.x + biVec.x);
			dqijKVec.y += lreVal * (biqVal * (2.0 / knlWidthSqu) * qjiVec.y + biVec.y);

			dqjjKVec.x += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qjiVec.x - bjVec.x);
			dqjjKVec.y += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qjiVec.y - bjVec.y);
		}

		setVector(d_dqijKMat, dqijKVec, lmkjIdx, lmkjNum);
		setVector(d_dqjjKMat, dqjjKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void d2qMatern2(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat, double *d_btmMat,
                           double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec, biVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);
		getVector(biVec, d_btmMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec, bjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);
			getVector(bjVec, d_btmMat, lmkjIdx, lmkNum);

			vector qijVec, bijVec;
			vectorSubtract(qijVec, qiVec, qjVec);
			vectorSubtract(bijVec, biVec, bjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -(p0Val + 2.0 * p1Val);
			double d2qKVal = p1Val;
			double  lrsVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                 / (8.0 * knlWidthSqu);
			double   bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lrsVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.x + d1qKVal * bijVec.x);
			dqKVec.y += lrsVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.y + d1qKVal * bijVec.y);
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void d2qiMatern2(double *d_dqiiKMat, double *d_dqjiKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqiiKVec = {0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0};

		vector qiVec, liVec, biVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
		getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec, bjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
			getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -(p0Val + 2.0 * p1Val);
			double d2qKVal = p1Val;
			double  lrsVal = dotProduct(liVec, rjVec) / (8.0 * knlWidthSqu);
			double  biqVal =  dotProduct(biVec, qijVec);
			double  bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.x + d1qKVal * biVec.x);
			dqiiKVec.y += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.y + d1qKVal * biVec.y);

			dqjiKVec.x += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.x - d1qKVal * bjVec.x);
			dqjiKVec.y += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.y - d1qKVal * bjVec.y);
		}

		setVector(d_dqiiKMat, dqiiKVec, lmkiIdx, lmkiNum);
		setVector(d_dqjiKMat, dqjiKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void d2qjMatern2(double *d_dqijKMat, double *d_dqjjKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqijKVec = {0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0};

		vector qjVec, rjVec, bjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
		getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec, biVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
			getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -(p0Val + 2.0 * p1Val);
			double d2qKVal = p1Val;
			double  lrsVal = dotProduct(liVec, rjVec) / (8.0 * knlWidthSqu);
			double  biqVal = -dotProduct(biVec, qjiVec);
			double  bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.x - d1qKVal * biVec.x);
			dqijKVec.y += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.y - d1qKVal * biVec.y);

			dqjjKVec.x += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.x + d1qKVal * bjVec.x);
			dqjjKVec.y += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.y + d1qKVal * bjVec.y);
		}

		setVector(d_dqijKMat, dqijKVec, lmkjIdx, lmkjNum);
		setVector(d_dqjjKMat, dqjjKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void d2qMatern3(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat, double *d_btmMat,
                           double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec, biVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);
		getVector(biVec, d_btmMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec, bjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);
			getVector(bjVec, d_btmMat, lmkjIdx, lmkNum);

			vector qijVec, bijVec;
			vectorSubtract(qijVec, qiVec, qjVec);
			vectorSubtract(bijVec, biVec, bjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -(4.0 * p0Val + (8.0 + dijSqu) * p1Val);
			double d2qKVal = p0Val + 2.0 * p1Val;
			double  lrsVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                 / (48.0 * knlWidthSqu);
			double   bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lrsVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.x + d1qKVal * bijVec.x);
			dqKVec.y += lrsVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.y + d1qKVal * bijVec.y);
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void d2qiMatern3(double *d_dqiiKMat, double *d_dqjiKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqiiKVec = {0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0};

		vector qiVec, liVec, biVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
		getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec, bjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
			getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -(4.0 * p0Val + (8.0 + dijSqu) * p1Val);
			double d2qKVal = p0Val + 2.0 * p1Val;
			double  lrsVal = dotProduct(liVec, rjVec) / (48.0 * knlWidthSqu);
			double  biqVal =  dotProduct(biVec, qijVec);
			double  bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.x + d1qKVal * biVec.x);
			dqiiKVec.y += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.y + d1qKVal * biVec.y);

			dqjiKVec.x += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.x - d1qKVal * bjVec.x);
			dqjiKVec.y += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.y - d1qKVal * bjVec.y);
		}

		setVector(d_dqiiKMat, dqiiKVec, lmkiIdx, lmkiNum);
		setVector(d_dqjiKMat, dqjiKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void d2qjMatern3(double *d_dqijKMat, double *d_dqjjKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqijKVec = {0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0};

		vector qjVec, rjVec, bjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
		getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec, biVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
			getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -(4.0 * p0Val + (8.0 + dijSqu) * p1Val);
			double d2qKVal = p0Val + 2.0 * p1Val;
			double  lrsVal = dotProduct(liVec, rjVec) / (48.0 * knlWidthSqu);
			double  biqVal = -dotProduct(biVec, qjiVec);
			double  bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.x - d1qKVal * biVec.x);
			dqijKVec.y += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.y - d1qKVal * biVec.y);

			dqjjKVec.x += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.x + d1qKVal * bjVec.x);
			dqjjKVec.y += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.y + d1qKVal * bjVec.y);
		}

		setVector(d_dqijKMat, dqijKVec, lmkjIdx, lmkjNum);
		setVector(d_dqjjKMat, dqjjKVec, lmkjIdx, lmkjNum);
	}

	return;
}

__global__ void d2qMatern4(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat, double *d_btmMat,
                           double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0};

		vector qiVec, liVec, riVec, biVec;
		getVector(qiVec, d_lmkMat, lmkiIdx, lmkNum);
		getVector(liVec, d_lftMat, lmkiIdx, lmkNum);
		getVector(riVec, d_rgtMat, lmkiIdx, lmkNum);
		getVector(biVec, d_btmMat, lmkiIdx, lmkNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
		{
			vector qjVec, ljVec, rjVec, bjVec;
			getVector(qjVec, d_lmkMat, lmkjIdx, lmkNum);
			getVector(ljVec, d_lftMat, lmkjIdx, lmkNum);
			getVector(rjVec, d_rgtMat, lmkjIdx, lmkNum);
			getVector(bjVec, d_btmMat, lmkjIdx, lmkNum);

			vector qijVec, bijVec;
			vectorSubtract(qijVec, qiVec, qjVec);
			vectorSubtract(bijVec, biVec, bjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
			double d2qKVal = 4.0 * p0Val + (8.0 + dijSqu) * p1Val;
			double  lrsVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                 / (384.0 * knlWidthSqu);
			double   bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lrsVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.x + d1qKVal * bijVec.x);
			dqKVec.y += lrsVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.y + d1qKVal * bijVec.y);
		}

		setVector(d_dqKMat, dqKVec, lmkiIdx, lmkNum);
	}

	return;
}

__global__ void d2qiMatern4(double *d_dqiiKMat, double *d_dqjiKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkiNum )
	{
		vector dqiiKVec = {0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0};

		vector qiVec, liVec, biVec;
		getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
		getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
		getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

		for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
		{
			vector qjVec, rjVec, bjVec;
			getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
			getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
			getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

			vector qijVec;
			vectorSubtract(qijVec, qiVec, qjVec);

			double dijVal = eucnorm(qijVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
			double d2qKVal = 4.0 * p0Val + (8.0 + dijSqu) * p1Val;
			double  lrsVal = dotProduct(liVec, rjVec) / (384.0 * knlWidthSqu);
			double  biqVal =  dotProduct(biVec, qijVec);
			double  bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.x + d1qKVal * biVec.x);
			dqiiKVec.y += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.y + d1qKVal * biVec.y);

			dqjiKVec.x += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.x - d1qKVal * bjVec.x);
			dqjiKVec.y += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.y - d1qKVal * bjVec.y);
		}

		setVector(d_dqiiKMat, dqiiKVec, lmkiIdx, lmkiNum);
		setVector(d_dqjiKMat, dqjiKVec, lmkiIdx, lmkiNum);
	}

	return;
}

__global__ void d2qjMatern4(double *d_dqijKMat, double *d_dqjjKMat, double *d_lmkiMat, double *d_lmkjMat,
                            double *d_lftMat, double *d_rgtMat, double *d_btmiMat, double *d_btmjMat,
                            double knlWidth, double knlWidthSqu, int lmkiNum, int lmkjNum)
{
	int lmkjIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkjIdx < lmkjNum )
	{
		vector dqijKVec = {0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0};

		vector qjVec, rjVec, bjVec;
		getVector(qjVec, d_lmkjMat, lmkjIdx, lmkjNum);
		getVector(rjVec, d_rgtMat,  lmkjIdx, lmkjNum);
		getVector(bjVec, d_btmjMat, lmkjIdx, lmkjNum);

		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			vector qiVec, liVec, biVec;
			getVector(qiVec, d_lmkiMat, lmkiIdx, lmkiNum);
			getVector(liVec, d_lftMat,  lmkiIdx, lmkiNum);
			getVector(biVec, d_btmiMat, lmkiIdx, lmkiNum);

			vector qjiVec;
			vectorSubtract(qjiVec, qjVec, qiVec);

			double dijVal = eucnorm(qjiVec) / knlWidth;
			double dijSqu = dijVal * dijVal;

			double p0Val, p1Val;
			p0Fcn(p0Val, dijVal);
			p1Fcn(p1Val, dijVal);

			double d1qKVal = -((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
			double d2qKVal = 4.0 * p0Val + (8.0 + dijSqu) * p1Val;
			double  lrsVal = dotProduct(liVec, rjVec) / (384.0 * knlWidthSqu);
			double  biqVal = -dotProduct(biVec, qjiVec);
			double  bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.x - d1qKVal * biVec.x);
			dqijKVec.y += lrsVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.y - d1qKVal * biVec.y);

			dqjjKVec.x += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.x + d1qKVal * bjVec.x);
			dqjjKVec.y += lrsVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.y + d1qKVal * bjVec.y);
		}

		setVector(d_dqijKMat, dqijKVec, lmkjIdx, lmkjNum);
		setVector(d_dqjjKMat, dqjjKVec, lmkjIdx, lmkjNum);
	}

	return;
}

void d2qKernel(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat, double *d_btmMat,
               int knlOrder, double knlWidth, int lmkNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

	double knlWidthSqu = knlWidth * knlWidth;

	int blkNum = (lmkNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			d2qGaussian <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, d_btmMat, 
			                                  knlWidth, knlWidthSqu, lmkNum);
			break;

		// Matern0 and Matern1 are not twice differentiable
	
		case 2:
			d2qMatern2 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, d_btmMat,
			                                 knlWidth, knlWidthSqu, lmkNum);
			break;
	
		case 3:
			d2qMatern3 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, d_btmMat,
			                                 knlWidth, knlWidthSqu, lmkNum);
			break;
	
		case 4:
			d2qMatern4 <<<blkNum, BLKDIM>>> (d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, d_btmMat,
			                                 knlWidth, knlWidthSqu, lmkNum);
			break;
	}	

	return;
}

void d2qKernel(double *d_dqiiKMat, double *d_dqijKMat, double *d_dqjiKMat, double *d_dqjjKMat,
               double *d_lmkiMat, double *d_lmkjMat, double *d_lftMat, double *d_rgtMat,
               double *d_btmiMat, double *d_btmjMat, int knlOrder, double knlWidth, int lmkiNum, int lmkjNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	setBesselkCoefficients();

	double knlWidthSqu = knlWidth * knlWidth;

	int blkiNum = (lmkiNum - 1) / BLKDIM + 1;
	int blkjNum = (lmkjNum - 1) / BLKDIM + 1;

	switch ( knlOrder )
	{
		case -1:
			d2qiGaussian <<<blkiNum, BLKDIM>>> (d_dqiiKMat, d_dqjiKMat, d_lmkiMat, d_lmkjMat,
			                                    d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                    knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			d2qjGaussian <<<blkjNum, BLKDIM>>> (d_dqijKMat, d_dqjjKMat, d_lmkiMat, d_lmkjMat,
			                                    d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                    knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;

		// Matern0 and Matern1 are not twice differentiable
	
		case 2:
			d2qiMatern2 <<<blkiNum, BLKDIM>>> (d_dqiiKMat, d_dqjiKMat, d_lmkiMat, d_lmkjMat,
			                                   d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			d2qjMatern2 <<<blkjNum, BLKDIM>>> (d_dqijKMat, d_dqjjKMat, d_lmkiMat, d_lmkjMat,
			                                   d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	
		case 3:
			d2qiMatern3 <<<blkiNum, BLKDIM>>> (d_dqiiKMat, d_dqjiKMat, d_lmkiMat, d_lmkjMat,
			                                   d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			d2qjMatern3 <<<blkjNum, BLKDIM>>> (d_dqijKMat, d_dqjjKMat, d_lmkiMat, d_lmkjMat,
			                                   d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	
		case 4:
			d2qiMatern4 <<<blkiNum, BLKDIM>>> (d_dqiiKMat, d_dqjiKMat, d_lmkiMat, d_lmkjMat,
			                                   d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			d2qjMatern4 <<<blkjNum, BLKDIM>>> (d_dqijKMat, d_dqjjKMat, d_lmkiMat, d_lmkjMat,
			                                   d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
			                                   knlWidth, knlWidthSqu, lmkiNum, lmkjNum);
			break;
	}	

	return;
}
