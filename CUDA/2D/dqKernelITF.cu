// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "constants.h"

void dqKernel(double *, double *, double *, double *, int, double, int);
void dqKernel(double *, double *, double *, double *, double *, double *, int, double, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	mxInitGPU();

	if ( nrhs == 5 )
	{
		mxGPUArray const *lmkMat, *lftMat, *rgtMat;
		mxGPUArray       *dqKMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkMat   = mxGPUCreateFromMxArray(prhs[0]);
		lftMat   = mxGPUCreateFromMxArray(prhs[1]);
		rgtMat   = mxGPUCreateFromMxArray(prhs[2]);
		knlOrder =            mxGetScalar(prhs[3]);
		knlWidth =            mxGetScalar(prhs[4]);
	
		if ( knlOrder == 0 )
			mexErrMsgIdAndTxt("dqKernel:order", "Matern kernel of order 0 is not differentiable.");
	
		mwSize const *lmkDims = mxGPUGetDimensions(lmkMat);
		int lmkNum = lmkDims[0];
	
		mwSize const ndim = 2;
		mwSize const dqKDims[2] = {(mwSize) lmkNum, (mwSize) DIMNUM};
		dqKMat = mxGPUCreateGPUArray(ndim, dqKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkMat = (double *) mxGPUGetDataReadOnly(lmkMat);
		double *d_lftMat = (double *) mxGPUGetDataReadOnly(lftMat);
		double *d_rgtMat = (double *) mxGPUGetDataReadOnly(rgtMat);
		double *d_dqKMat = (double *) mxGPUGetData(dqKMat);
	
		// ---
	
		dqKernel(d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, knlOrder, knlWidth, lmkNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(dqKMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkMat);
		mxGPUDestroyGPUArray(lftMat);
		mxGPUDestroyGPUArray(rgtMat);
		mxGPUDestroyGPUArray(dqKMat);
	
		mxFree((void *) lmkDims);
	}
	else if ( nrhs == 6 )
	{
		mxGPUArray const *lmkiMat, *lmkjMat, *lftMat, *rgtMat;
		mxGPUArray       *dqiKMat, *dqjKMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkiMat  = mxGPUCreateFromMxArray(prhs[0]);
		lmkjMat  = mxGPUCreateFromMxArray(prhs[1]);
		lftMat   = mxGPUCreateFromMxArray(prhs[2]);
		rgtMat   = mxGPUCreateFromMxArray(prhs[3]);
		knlOrder =            mxGetScalar(prhs[4]);
		knlWidth =            mxGetScalar(prhs[5]);
	
		if ( knlOrder == 0 )
			mexErrMsgIdAndTxt("dqKernel:order", "Matern kernel of order 0 is not differentiable.");
	
		mwSize const *lmkiDims = mxGPUGetDimensions(lmkiMat);
		mwSize const *lmkjDims = mxGPUGetDimensions(lmkjMat);
		int lmkiNum = lmkiDims[0];
		int lmkjNum = lmkjDims[0];
	
		mwSize const ndim = 2;
		mwSize const dqiKDims[2] = {(mwSize) lmkiNum, (mwSize) DIMNUM};
		mwSize const dqjKDims[2] = {(mwSize) lmkjNum, (mwSize) DIMNUM};
		dqiKMat = mxGPUCreateGPUArray(ndim, dqiKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		dqjKMat = mxGPUCreateGPUArray(ndim, dqjKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkiMat = (double *) mxGPUGetDataReadOnly(lmkiMat);
		double *d_lmkjMat = (double *) mxGPUGetDataReadOnly(lmkjMat);
		double *d_lftMat  = (double *) mxGPUGetDataReadOnly(lftMat);
		double *d_rgtMat  = (double *) mxGPUGetDataReadOnly(rgtMat);
		double *d_dqiKMat = (double *) mxGPUGetData(dqiKMat);
		double *d_dqjKMat = (double *) mxGPUGetData(dqjKMat);
	
		// ---
	
		dqKernel(d_dqiKMat, d_dqjKMat, d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat,
		         knlOrder, knlWidth, lmkiNum, lmkjNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(dqiKMat);
		plhs[1] = mxGPUCreateMxArrayOnGPU(dqjKMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkiMat);
		mxGPUDestroyGPUArray(lmkjMat);
		mxGPUDestroyGPUArray(lftMat);
		mxGPUDestroyGPUArray(rgtMat);
		mxGPUDestroyGPUArray(dqiKMat);
		mxGPUDestroyGPUArray(dqjKMat);
	
		mxFree((void *) lmkiDims);
		mxFree((void *) lmkjDims);
	}
	else
		mexErrMsgIdAndTxt("dqKernel:nrhs", "The number of inputs must be 5 or 6.");

	return;
}

