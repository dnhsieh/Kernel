// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cstring>
#include <cmath>
#include "mex.h"

void d2qKernel(double *, double *, double *, double *, double *, int, double, int, int);
void d2qKernel(double *, double *, double *, double *,
               double *, double *, double *, double *, double *, double *,
               int, double, int, int, int);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	if ( nrhs == 6 )
	{
		double *lmkMat, *lftMat, *rgtMat, *btmMat;
		int     knlOrder;
		double  knlWidth;
	
		double *dqKMat;
	
		lmkMat   = mxGetDoubles(prhs[0]);
		lftMat   = mxGetDoubles(prhs[1]);
		rgtMat   = mxGetDoubles(prhs[2]);
		btmMat   = mxGetDoubles(prhs[3]);
		knlOrder =  mxGetScalar(prhs[4]);
		knlWidth =  mxGetScalar(prhs[5]);
	
		if ( knlOrder == 0 || knlOrder == 1 )
		{
			mexErrMsgIdAndTxt("d2qKernel:order", 
			                  "Matern kernel of order %d is not twice differentiable.", knlOrder);
		}
	
		int dimNum = mxGetM(prhs[0]);
		int lmkNum = mxGetN(prhs[0]);
	
		plhs[0] = mxCreateDoubleMatrix(dimNum, lmkNum, mxREAL);
		dqKMat  = mxGetDoubles(plhs[0]);
	
		d2qKernel(dqKMat, lmkMat, lftMat, rgtMat, btmMat, knlOrder, knlWidth, lmkNum, dimNum);
	}
	else if ( nrhs == 8 )
	{
		double *lmkiMat, *lmkjMat, *lftMat, *rgtMat, *btmiMat, *btmjMat;
		int     knlOrder;
		double  knlWidth;
	
		double *dqiiKMat, *dqijKMat, *dqjiKMat, *dqjjKMat;
	
		lmkiMat  = mxGetDoubles(prhs[0]);
		lmkjMat  = mxGetDoubles(prhs[1]);
		lftMat   = mxGetDoubles(prhs[2]);
		rgtMat   = mxGetDoubles(prhs[3]);
		btmiMat  = mxGetDoubles(prhs[4]);
		btmjMat  = mxGetDoubles(prhs[5]);
		knlOrder =  mxGetScalar(prhs[6]);
		knlWidth =  mxGetScalar(prhs[7]);
	
		if ( knlOrder == 0 || knlOrder == 1 )
		{
			mexErrMsgIdAndTxt("d2qKernel:order", 
			                  "Matern kernel of order %d is not twice differentiable.", knlOrder);
		}
	
		int  dimNum = mxGetM(prhs[0]);
		int lmkiNum = mxGetN(prhs[0]);
		int lmkjNum = mxGetN(prhs[1]);
	
		plhs[0]  = mxCreateDoubleMatrix(dimNum, lmkiNum, mxREAL);
		plhs[1]  = mxCreateDoubleMatrix(dimNum, lmkjNum, mxREAL);
		plhs[2]  = mxCreateDoubleMatrix(dimNum, lmkiNum, mxREAL);
		plhs[3]  = mxCreateDoubleMatrix(dimNum, lmkjNum, mxREAL);
		dqiiKMat = mxGetDoubles(plhs[0]);
		dqijKMat = mxGetDoubles(plhs[1]);
		dqjiKMat = mxGetDoubles(plhs[2]);
		dqjjKMat = mxGetDoubles(plhs[3]);
	
		d2qKernel(dqiiKMat, dqijKMat, dqjiKMat, dqjjKMat,
		          lmkiMat, lmkjMat, lftMat, rgtMat, btmiMat, btmjMat,
		          knlOrder, knlWidth, lmkiNum, lmkjNum, dimNum);
	}
	else
		mexErrMsgIdAndTxt("d2qKernel:nrhs", "The number of inputs must be 6 or 8.");

	return;
}
