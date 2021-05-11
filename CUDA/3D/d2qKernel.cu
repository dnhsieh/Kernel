// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include "matvec.h"
#include "constants.h"

__global__ void d2qGaussian(double *d_dqKMat, double *d_lmkMat, double *d_lftMat, double *d_rgtMat, double *d_btmMat,
                            double knlWidth, double knlWidthSqu, int lmkNum)
{
	int lmkiIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( lmkiIdx < lmkNum )
	{
		vector dqKVec = {0.0, 0.0, 0.0};

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
			dqKVec.z += lreVal * (bqVal * (2.0 / knlWidthSqu) * qijVec.z - bijVec.z);
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
		vector dqiiKVec = {0.0, 0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0, 0.0};

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
			dqiiKVec.z += lreVal * (biqVal * (2.0 / knlWidthSqu) * qijVec.z - biVec.z);

			dqjiKVec.x += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qijVec.x + bjVec.x);
			dqjiKVec.y += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qijVec.y + bjVec.y);
			dqjiKVec.z += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qijVec.z + bjVec.z);
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
		vector dqijKVec = {0.0, 0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0, 0.0};

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
			dqijKVec.z += lreVal * (biqVal * (2.0 / knlWidthSqu) * qjiVec.z + biVec.z);

			dqjjKVec.x += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qjiVec.x - bjVec.x);
			dqjjKVec.y += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qjiVec.y - bjVec.y);
			dqjjKVec.z += lreVal * (bjqVal * (2.0 / knlWidthSqu) * qjiVec.z - bjVec.z);
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
		vector dqKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qijVec) / knlWidth;
			double d1qKVal = -(1.0 + dijVal);
			double  lreVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                 * exp(-dijVal) / (3.0 * knlWidthSqu);
			double   bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lreVal * (bqVal / knlWidthSqu * qijVec.x + d1qKVal * bijVec.x);
			dqKVec.y += lreVal * (bqVal / knlWidthSqu * qijVec.y + d1qKVal * bijVec.y);
			dqKVec.z += lreVal * (bqVal / knlWidthSqu * qijVec.z + d1qKVal * bijVec.z);
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
		vector dqiiKVec = {0.0, 0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qijVec) / knlWidth;
			double d1qKVal = -(1.0 + dijVal);
			double  lreVal = dotProduct(liVec, rjVec) * exp(-dijVal) / (3.0 * knlWidthSqu);
			double  biqVal =  dotProduct(biVec, qijVec);
			double  bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lreVal * (biqVal / knlWidthSqu * qijVec.x + d1qKVal * biVec.x);
			dqiiKVec.y += lreVal * (biqVal / knlWidthSqu * qijVec.y + d1qKVal * biVec.y);
			dqiiKVec.z += lreVal * (biqVal / knlWidthSqu * qijVec.z + d1qKVal * biVec.z);

			dqjiKVec.x += lreVal * (bjqVal / knlWidthSqu * qijVec.x - d1qKVal * bjVec.x);
			dqjiKVec.y += lreVal * (bjqVal / knlWidthSqu * qijVec.y - d1qKVal * bjVec.y);
			dqjiKVec.z += lreVal * (bjqVal / knlWidthSqu * qijVec.z - d1qKVal * bjVec.z);
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
		vector dqijKVec = {0.0, 0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qjiVec) / knlWidth;
			double d1qKVal = -(1.0 + dijVal);
			double  lreVal = dotProduct(liVec, rjVec) * exp(-dijVal) / (3.0 * knlWidthSqu);
			double  biqVal = -dotProduct(biVec, qjiVec);
			double  bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lreVal * (biqVal / knlWidthSqu * qjiVec.x - d1qKVal * biVec.x);
			dqijKVec.y += lreVal * (biqVal / knlWidthSqu * qjiVec.y - d1qKVal * biVec.y);
			dqijKVec.z += lreVal * (biqVal / knlWidthSqu * qjiVec.z - d1qKVal * biVec.z);

			dqjjKVec.x += lreVal * (bjqVal / knlWidthSqu * qjiVec.x + d1qKVal * bjVec.x);
			dqjjKVec.y += lreVal * (bjqVal / knlWidthSqu * qjiVec.y + d1qKVal * bjVec.y);
			dqjjKVec.z += lreVal * (bjqVal / knlWidthSqu * qjiVec.z + d1qKVal * bjVec.z);
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
		vector dqKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qijVec) / knlWidth;
			double d1qKVal = -(3.0 + dijVal * (3.0 + dijVal));
			double d2qKVal = 1.0 + dijVal;
			double  lreVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                 * exp(-dijVal) / (15.0 * knlWidthSqu);
			double   bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lreVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.x + d1qKVal * bijVec.x);
			dqKVec.y += lreVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.y + d1qKVal * bijVec.y);
			dqKVec.z += lreVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.z + d1qKVal * bijVec.z);
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
		vector dqiiKVec = {0.0, 0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qijVec) / knlWidth;
			double d1qKVal = -(3.0 + dijVal * (3.0 + dijVal));
			double d2qKVal = 1.0 + dijVal;
			double  lreVal = dotProduct(liVec, rjVec) * exp(-dijVal) / (15.0 * knlWidthSqu);
			double  biqVal =  dotProduct(biVec, qijVec);
			double  bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lreVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.x + d1qKVal * biVec.x);
			dqiiKVec.y += lreVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.y + d1qKVal * biVec.y);
			dqiiKVec.z += lreVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.z + d1qKVal * biVec.z);

			dqjiKVec.x += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.x - d1qKVal * bjVec.x);
			dqjiKVec.y += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.y - d1qKVal * bjVec.y);
			dqjiKVec.z += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.z - d1qKVal * bjVec.z);
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
		vector dqijKVec = {0.0, 0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qjiVec) / knlWidth;
			double d1qKVal = -(3.0 + dijVal * (3.0 + dijVal));
			double d2qKVal = 1.0 + dijVal;
			double  lreVal = dotProduct(liVec, rjVec) * exp(-dijVal) / (15.0 * knlWidthSqu);
			double  biqVal = -dotProduct(biVec, qjiVec);
			double  bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lreVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.x - d1qKVal * biVec.x);
			dqijKVec.y += lreVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.y - d1qKVal * biVec.y);
			dqijKVec.z += lreVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.z - d1qKVal * biVec.z);

			dqjjKVec.x += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.x + d1qKVal * bjVec.x);
			dqjjKVec.y += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.y + d1qKVal * bjVec.y);
			dqjjKVec.z += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.z + d1qKVal * bjVec.z);
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
		vector dqKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qijVec) / knlWidth;
			double d1qKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal)));
			double d2qKVal = 3.0 + dijVal * (3.0 + dijVal);
			double  lreVal = (dotProduct(liVec, rjVec) + dotProduct(ljVec, riVec))
			                 * exp(-dijVal) / (105.0 * knlWidthSqu);
			double   bqVal = dotProduct(bijVec, qijVec);

			dqKVec.x += lreVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.x + d1qKVal * bijVec.x);
			dqKVec.y += lreVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.y + d1qKVal * bijVec.y);
			dqKVec.z += lreVal * (d2qKVal * bqVal / knlWidthSqu * qijVec.z + d1qKVal * bijVec.z);
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
		vector dqiiKVec = {0.0, 0.0, 0.0};
		vector dqjiKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qijVec) / knlWidth;
			double d1qKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal)));
			double d2qKVal = 3.0 + dijVal * (3.0 + dijVal);
			double  lreVal = dotProduct(liVec, rjVec) * exp(-dijVal) / (105.0 * knlWidthSqu);
			double  biqVal =  dotProduct(biVec, qijVec);
			double  bjqVal = -dotProduct(bjVec, qijVec);

			dqiiKVec.x += lreVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.x + d1qKVal * biVec.x);
			dqiiKVec.y += lreVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.y + d1qKVal * biVec.y);
			dqiiKVec.z += lreVal * (d2qKVal * biqVal / knlWidthSqu * qijVec.z + d1qKVal * biVec.z);

			dqjiKVec.x += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.x - d1qKVal * bjVec.x);
			dqjiKVec.y += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.y - d1qKVal * bjVec.y);
			dqjiKVec.z += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qijVec.z - d1qKVal * bjVec.z);
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
		vector dqijKVec = {0.0, 0.0, 0.0};
		vector dqjjKVec = {0.0, 0.0, 0.0};

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

			double  dijVal = eucnorm(qjiVec) / knlWidth;
			double d1qKVal = -(15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal)));
			double d2qKVal = 3.0 + dijVal * (3.0 + dijVal);
			double  lreVal = dotProduct(liVec, rjVec) * exp(-dijVal) / (105.0 * knlWidthSqu);
			double  biqVal = -dotProduct(biVec, qjiVec);
			double  bjqVal =  dotProduct(bjVec, qjiVec);

			dqijKVec.x += lreVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.x - d1qKVal * biVec.x);
			dqijKVec.y += lreVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.y - d1qKVal * biVec.y);
			dqijKVec.z += lreVal * (d2qKVal * biqVal / knlWidthSqu * qjiVec.z - d1qKVal * biVec.z);

			dqjjKVec.x += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.x + d1qKVal * bjVec.x);
			dqjjKVec.y += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.y + d1qKVal * bjVec.y);
			dqjjKVec.z += lreVal * (d2qKVal * bjqVal / knlWidthSqu * qjiVec.z + d1qKVal * bjVec.z);
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
