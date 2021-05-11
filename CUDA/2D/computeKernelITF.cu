// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include "mex.h"
#include "gpu/mxGPUArray.h"

void computeKernel(double *, double *, int, double, int);
void computeKernel(double *, double *, double *, int, double, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	mxInitGPU();

	if ( nrhs == 3 )
	{
		mxGPUArray const *lmkMat;
		mxGPUArray       *knlMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkMat   = mxGPUCreateFromMxArray(prhs[0]);
		knlOrder =            mxGetScalar(prhs[1]);
		knlWidth =            mxGetScalar(prhs[2]);
	
		mwSize const *lmkDims = mxGPUGetDimensions(lmkMat);
		int lmkNum = lmkDims[0];
	
		mwSize const ndim = 2;
		mwSize const knlDims[2] = {(mwSize) lmkNum, (mwSize) lmkNum};
		knlMat = mxGPUCreateGPUArray(ndim, knlDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkMat = (double *) mxGPUGetDataReadOnly(lmkMat);
		double *d_knlMat = (double *) mxGPUGetData(knlMat);
	
		// ---
	
		computeKernel(d_knlMat, d_lmkMat, knlOrder, knlWidth, lmkNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(knlMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkMat);
		mxGPUDestroyGPUArray(knlMat);
	
		mxFree((void *) lmkDims);
	}
	else if ( nrhs == 4 )
	{
		mxGPUArray const *lmkiMat, *lmkjMat;
		mxGPUArray       *knlMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkiMat  = mxGPUCreateFromMxArray(prhs[0]);
		lmkjMat  = mxGPUCreateFromMxArray(prhs[1]);
		knlOrder =            mxGetScalar(prhs[2]);
		knlWidth =            mxGetScalar(prhs[3]);
	
		mwSize const *lmkiDims = mxGPUGetDimensions(lmkiMat);
		mwSize const *lmkjDims = mxGPUGetDimensions(lmkjMat);
		int lmkiNum = lmkiDims[0];
		int lmkjNum = lmkjDims[0];
	
		mwSize const ndim = 2;
		mwSize const knlDims[2] = {(mwSize) lmkiNum, (mwSize) lmkjNum};
		knlMat = mxGPUCreateGPUArray(ndim, knlDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkiMat = (double *) mxGPUGetDataReadOnly(lmkiMat);
		double *d_lmkjMat = (double *) mxGPUGetDataReadOnly(lmkjMat);
		double *d_knlMat  = (double *) mxGPUGetData(knlMat);
	
		// ---
	
		computeKernel(d_knlMat, d_lmkiMat, d_lmkjMat, knlOrder, knlWidth, lmkiNum, lmkjNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(knlMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkiMat);
		mxGPUDestroyGPUArray(lmkjMat);
		mxGPUDestroyGPUArray(knlMat);
	
		mxFree((void *) lmkiDims);
		mxFree((void *) lmkjDims);
	}
	else
		mexErrMsgIdAndTxt("computeKernel:nrhs", "The number of inputs must be 3 or 4.");

	return;
}

