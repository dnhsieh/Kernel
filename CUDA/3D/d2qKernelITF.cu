// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/18/2020

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "constants.h"

void d2qKernel(double *, double *, double *, double *, double *, int, double, int);
void d2qKernel(double *, double *, double *, double *,
               double *, double *, double *, double *, double *, double *,
               int, double, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	mxInitGPU();

	if ( nrhs == 6 )
	{
		mxGPUArray const *lmkMat, *lftMat, *rgtMat, *btmMat;
		mxGPUArray       *dqKMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkMat   = mxGPUCreateFromMxArray(prhs[0]);
		lftMat   = mxGPUCreateFromMxArray(prhs[1]);
		rgtMat   = mxGPUCreateFromMxArray(prhs[2]);
		btmMat   = mxGPUCreateFromMxArray(prhs[3]);
		knlOrder =            mxGetScalar(prhs[4]);
		knlWidth =            mxGetScalar(prhs[5]);
	
		if ( knlOrder == 0 || knlOrder == 1 )
		{
			mexErrMsgIdAndTxt("d2qKernel:order", 
			                  "Matern kernel of order %d is not twice differentiable.", knlOrder);
		}
	
		mwSize const *lmkDims = mxGPUGetDimensions(lmkMat);
		int lmkNum = lmkDims[0];
	
		mwSize const ndim = 2;
		mwSize const dqKDims[2] = {(mwSize) lmkNum, (mwSize) DIMNUM};
		dqKMat = mxGPUCreateGPUArray(ndim, dqKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkMat = (double *) mxGPUGetDataReadOnly(lmkMat);
		double *d_lftMat = (double *) mxGPUGetDataReadOnly(lftMat);
		double *d_rgtMat = (double *) mxGPUGetDataReadOnly(rgtMat);
		double *d_btmMat = (double *) mxGPUGetDataReadOnly(btmMat);
		double *d_dqKMat = (double *) mxGPUGetData(dqKMat);
	
		// ---
	
		d2qKernel(d_dqKMat, d_lmkMat, d_lftMat, d_rgtMat, d_btmMat, knlOrder, knlWidth, lmkNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(dqKMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkMat);
		mxGPUDestroyGPUArray(lftMat);
		mxGPUDestroyGPUArray(rgtMat);
		mxGPUDestroyGPUArray(btmMat);
		mxGPUDestroyGPUArray(dqKMat);
	
		mxFree((void *) lmkDims);
	}
	else if ( nrhs == 8 )
	{
		mxGPUArray const *lmkiMat, *lmkjMat, *lftMat, *rgtMat, *btmiMat, *btmjMat;
		mxGPUArray       *dqiiKMat, *dqijKMat, *dqjiKMat, *dqjjKMat;
	
		int    knlOrder;
		double knlWidth;
	
		lmkiMat  = mxGPUCreateFromMxArray(prhs[0]);
		lmkjMat  = mxGPUCreateFromMxArray(prhs[1]);
		lftMat   = mxGPUCreateFromMxArray(prhs[2]);
		rgtMat   = mxGPUCreateFromMxArray(prhs[3]);
		btmiMat  = mxGPUCreateFromMxArray(prhs[4]);
		btmjMat  = mxGPUCreateFromMxArray(prhs[5]);
		knlOrder =            mxGetScalar(prhs[6]);
		knlWidth =            mxGetScalar(prhs[7]);
	
		if ( knlOrder == 0 || knlOrder == 1 )
		{
			mexErrMsgIdAndTxt("d2qKernel:order", 
			                  "Matern kernel of order %d is not twice differentiable.", knlOrder);
		}
	
		mwSize const *lmkiDims = mxGPUGetDimensions(lmkiMat);
		mwSize const *lmkjDims = mxGPUGetDimensions(lmkjMat);
		int lmkiNum = lmkiDims[0];
		int lmkjNum = lmkjDims[0];
	
		mwSize const ndim = 2;
		mwSize const dqiiKDims[2] = {(mwSize) lmkiNum, (mwSize) DIMNUM};
		mwSize const dqijKDims[2] = {(mwSize) lmkjNum, (mwSize) DIMNUM};
		mwSize const dqjiKDims[2] = {(mwSize) lmkiNum, (mwSize) DIMNUM};
		mwSize const dqjjKDims[2] = {(mwSize) lmkjNum, (mwSize) DIMNUM};
		dqiiKMat = mxGPUCreateGPUArray(ndim, dqiiKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		dqijKMat = mxGPUCreateGPUArray(ndim, dqijKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		dqjiKMat = mxGPUCreateGPUArray(ndim, dqjiKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
		dqjjKMat = mxGPUCreateGPUArray(ndim, dqjjKDims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
	
		// ---
	
		double *d_lmkiMat  = (double *) mxGPUGetDataReadOnly(lmkiMat);
		double *d_lmkjMat  = (double *) mxGPUGetDataReadOnly(lmkjMat);
		double *d_lftMat   = (double *) mxGPUGetDataReadOnly(lftMat );
		double *d_rgtMat   = (double *) mxGPUGetDataReadOnly(rgtMat );
		double *d_btmiMat  = (double *) mxGPUGetDataReadOnly(btmiMat);
		double *d_btmjMat  = (double *) mxGPUGetDataReadOnly(btmjMat);
		double *d_dqiiKMat = (double *) mxGPUGetData(dqiiKMat);
		double *d_dqijKMat = (double *) mxGPUGetData(dqijKMat);
		double *d_dqjiKMat = (double *) mxGPUGetData(dqjiKMat);
		double *d_dqjjKMat = (double *) mxGPUGetData(dqjjKMat);
	
		// ---
	
		d2qKernel(d_dqiiKMat, d_dqijKMat, d_dqjiKMat, d_dqjjKMat,
		          d_lmkiMat, d_lmkjMat, d_lftMat, d_rgtMat, d_btmiMat, d_btmjMat,
		          knlOrder, knlWidth, lmkiNum, lmkjNum);
		plhs[0] = mxGPUCreateMxArrayOnGPU(dqiiKMat);
		plhs[1] = mxGPUCreateMxArrayOnGPU(dqijKMat);
		plhs[2] = mxGPUCreateMxArrayOnGPU(dqjiKMat);
		plhs[3] = mxGPUCreateMxArrayOnGPU(dqjjKMat);
	
		// ---
	
		mxGPUDestroyGPUArray(lmkiMat);
		mxGPUDestroyGPUArray(lmkjMat);
		mxGPUDestroyGPUArray(lftMat);
		mxGPUDestroyGPUArray(rgtMat);
		mxGPUDestroyGPUArray(btmiMat);
		mxGPUDestroyGPUArray(btmjMat);
		mxGPUDestroyGPUArray(dqiiKMat);
		mxGPUDestroyGPUArray(dqijKMat);
		mxGPUDestroyGPUArray(dqjiKMat);
		mxGPUDestroyGPUArray(dqjjKMat);
	
		mxFree((void *) lmkiDims);
		mxFree((void *) lmkjDims);
	}
	else
		mexErrMsgIdAndTxt("d2qKernel:nrhs", "The number of inputs must be 6 or 8.");

	return;
}

