// Author: Dai-Ni Hsieh (dnhsieh@jhu.edu)
// Date  : 11/17/2020

#include <cstring>
#include <cmath>

void p0Fcn(double &p0Val, double xVal);
void p1Fcn(double &p1Val, double xVal);

double eucdist(double *v1Vec, double *v2Vec, int dimNum)
{
	double dstSqu = 0.0;
	for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
	{
		double difVal = v1Vec[dimIdx] - v2Vec[dimIdx];
		dstSqu += difVal * difVal;
	}

	return sqrt(dstSqu);
}

double dotProduct(double *v1Vec, double *v2Vec, int dimNum)
{
	double dotVal = 0.0;
	for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
		dotVal += v1Vec[dimIdx] * v2Vec[dimIdx];

	return dotVal;
}

void d2qKernel(double *dqKMat, double *lmkMat, double *lftMat, double *rgtMat, double *btmMat,
               int knlOrder, double knlWidth, int lmkNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	double knlWidthSqu = knlWidth * knlWidth;

	memset(dqKMat, 0, sizeof(double) * dimNum * lmkNum);

	if ( knlOrder == -1 )   // Gaussian
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
		{
			double   *qiVec = lmkMat + lmkiIdx * dimNum;
			double   *liVec = lftMat + lmkiIdx * dimNum;
			double   *riVec = rgtMat + lmkiIdx * dimNum;
			double   *biVec = btmMat + lmkiIdx * dimNum;
			double *dqiKVec = dqKMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
			{
				double *qjVec = lmkMat + lmkjIdx * dimNum;
				double *ljVec = lftMat + lmkjIdx * dimNum;
				double *rjVec = rgtMat + lmkjIdx * dimNum;
				double *bjVec = btmMat + lmkjIdx * dimNum;

				double bqVal = 0.0;
				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
				{
					bqVal +=  (biVec[dimIdx] - bjVec[dimIdx])
					        * (qiVec[dimIdx] - qjVec[dimIdx]);
				}

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double lreVal = (  dotProduct(liVec, rjVec, dimNum)
				                 + dotProduct(ljVec, riVec, dimNum) )
				                * 2.0 * exp(-dijVal * dijVal) / knlWidthSqu;
				
				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
				{
					dqiKVec[dimIdx] += 
					   lreVal
					   *
					   (
					     bqVal * 2.0 / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
						  -
					     (biVec[dimIdx] - bjVec[dimIdx])
					   );
				}
			}
		}
	
	}
	else   // Matern
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double   *biVec = btmMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;
							double *bjVec = btmMat + lmkjIdx * dimNum;

							double bqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								bqVal +=   (biVec[dimIdx] - bjVec[dimIdx])
								         * (qiVec[dimIdx] - qjVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double d1qKVal = -(p0Val + 2.0 * p1Val);
							double d2qKVal = p1Val;
							double  lrsVal = (  dotProduct(liVec, rjVec, dimNum)
							                  + dotProduct(ljVec, riVec, dimNum) )
							                 / (8.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * bqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     d1qKVal * (biVec[dimIdx] - bjVec[dimIdx])
								   );
							}
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double   *biVec = btmMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;
							double *bjVec = btmMat + lmkjIdx * dimNum;

							double bqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								bqVal +=   (biVec[dimIdx] - bjVec[dimIdx])
								         * (qiVec[dimIdx] - qjVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double d1qKVal = -(4.0 * p0Val + (8.0 + dijSqu) * p1Val);
							double d2qKVal = p0Val + 2.0 * p1Val;
							double  lrsVal = (  dotProduct(liVec, rjVec, dimNum)
							                  + dotProduct(ljVec, riVec, dimNum) )
							                 / (48.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * bqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     d1qKVal * (biVec[dimIdx] - bjVec[dimIdx])
								   );
							}
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double   *biVec = btmMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;
							double *bjVec = btmMat + lmkjIdx * dimNum;

							double bqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								bqVal +=   (biVec[dimIdx] - bjVec[dimIdx])
								         * (qiVec[dimIdx] - qjVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double d1qKVal = -((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
							double d2qKVal = 4.0 * p0Val + (8.0 + dijSqu) * p1Val;
							double  lrsVal = (  dotProduct(liVec, rjVec, dimNum)
							                  + dotProduct(ljVec, riVec, dimNum) )
							                 / (384.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * bqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     d1qKVal * (biVec[dimIdx] - bjVec[dimIdx])
								   );
							}
						}
					}
					break;
			}
		}
		else   // dimNum == 3
		{
			switch ( knlOrder )
			{
				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double   *biVec = btmMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;
							double *bjVec = btmMat + lmkjIdx * dimNum;

							double bqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								bqVal +=   (biVec[dimIdx] - bjVec[dimIdx])
								         * (qiVec[dimIdx] - qjVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double lreVal = (  dotProduct(liVec, rjVec, dimNum)
							                 + dotProduct(ljVec, riVec, dimNum) )
							                * exp(-dijVal) / (3.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     bqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     (1.0 + dijVal) * (biVec[dimIdx] - bjVec[dimIdx])
								   );
							}
						}
					}
					break;
			
				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double   *biVec = btmMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;
							double *bjVec = btmMat + lmkjIdx * dimNum;

							double bqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								bqVal +=   (biVec[dimIdx] - bjVec[dimIdx])
								         * (qiVec[dimIdx] - qjVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double lreVal = (  dotProduct(liVec, rjVec, dimNum)
							                 + dotProduct(ljVec, riVec, dimNum) )
							                * exp(-dijVal) / (15.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     bqVal * (1.0 + dijVal) / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     (3.0 + dijVal * (3.0 + dijVal)) * (biVec[dimIdx] - bjVec[dimIdx])
								   );
							}
						}
					}
					break;
			
				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkNum; ++lmkiIdx )
					{
						double   *qiVec = lmkMat + lmkiIdx * dimNum;
						double   *liVec = lftMat + lmkiIdx * dimNum;
						double   *riVec = rgtMat + lmkiIdx * dimNum;
						double   *biVec = btmMat + lmkiIdx * dimNum;
						double *dqiKVec = dqKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkNum; ++lmkjIdx )
						{
							double *qjVec = lmkMat + lmkjIdx * dimNum;
							double *ljVec = lftMat + lmkjIdx * dimNum;
							double *rjVec = rgtMat + lmkjIdx * dimNum;
							double *bjVec = btmMat + lmkjIdx * dimNum;

							double bqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								bqVal +=   (biVec[dimIdx] - bjVec[dimIdx])
								         * (qiVec[dimIdx] - qjVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double lreVal = (  dotProduct(liVec, rjVec, dimNum)
							                 + dotProduct(ljVec, riVec, dimNum) )
							                * exp(-dijVal) / (105.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     bqVal * (3.0 + dijVal * (3.0 + dijVal)) / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * (biVec[dimIdx] - bjVec[dimIdx])
								   );
							}
						}
					}
					break;
			}
		}
	}

	return;
}

void d2qKernel(double *dqiiKMat, double *dqijKMat, double *dqjiKMat, double *dqjjKMat,
               double *lmkiMat, double *lmkjMat, double *lftMat, double *rgtMat, double *btmiMat, double *btmjMat,
               int knlOrder, double knlWidth, int lmkiNum, int lmkjNum, int dimNum)
{
	// order 0 to 4: Matern kernel of order 0 to 4
	// order     -1: Gaussian kernel

	double knlWidthSqu = knlWidth * knlWidth;

	memset(dqiiKMat, 0, sizeof(double) * dimNum * lmkiNum);
	memset(dqijKMat, 0, sizeof(double) * dimNum * lmkjNum);
	memset(dqjiKMat, 0, sizeof(double) * dimNum * lmkiNum);
	memset(dqjjKMat, 0, sizeof(double) * dimNum * lmkjNum);

	if ( knlOrder == -1 )   // Gaussian
	{
		for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
		{
			double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
			double    *liVec =   lftMat + lmkiIdx * dimNum;
			double    *biVec =  btmiMat + lmkiIdx * dimNum;
			double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
			double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

			for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
			{
				double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
				double    *rjVec =   rgtMat + lmkjIdx * dimNum;
				double    *bjVec =  btmjMat + lmkjIdx * dimNum;
				double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
				double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

				double biqVal = 0.0, bjqVal = 0.0;
				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
				{
					biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
					bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
				}

				double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
				double lreVal = dotProduct(liVec, rjVec, dimNum) * 2.0 * exp(-dijVal * dijVal) / knlWidthSqu;
				
				for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
				{
					dqiiKVec[dimIdx] += 
					   lreVal
					   *
					   (
					     biqVal * 2.0 / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
						  -
					     biVec[dimIdx]
					   );

					dqijKVec[dimIdx] += 
					   lreVal
					   *
					   (
					     -biqVal * 2.0 / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
						  +
					     biVec[dimIdx]
					   );

					dqjiKVec[dimIdx] += 
					   lreVal
					   *
					   (
					     -bjqVal * 2.0 / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
						  +
					     bjVec[dimIdx]
					   );

					dqjjKVec[dimIdx] += 
					   lreVal
					   *
					   (
					     bjqVal * 2.0 / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
						  -
					     bjVec[dimIdx]
					   );
				}
			}
		}
	
	}
	else   // Matern
	{
		if ( dimNum == 2 )
		{
			switch ( knlOrder )
			{
				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
						double    *liVec =   lftMat + lmkiIdx * dimNum;
						double    *biVec =  btmiMat + lmkiIdx * dimNum;
						double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
						double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
							double    *rjVec =   rgtMat + lmkjIdx * dimNum;
							double    *bjVec =  btmjMat + lmkjIdx * dimNum;
							double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
							double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

							double biqVal = 0.0, bjqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
								bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double d1qKVal = -(p0Val + 2.0 * p1Val);
							double d2qKVal = p1Val;
							double  lrsVal = dotProduct(liVec, rjVec, dimNum) / (8.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     d1qKVal * biVec[dimIdx]
								   );

								dqijKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     -d2qKVal * biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     d1qKVal * biVec[dimIdx]
								   );

								dqjiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     -d2qKVal * bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  -
								     d1qKVal * bjVec[dimIdx]
								   );

								dqjjKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  +
								     d1qKVal * bjVec[dimIdx]
								   );
							}
						}
					}
					break;

				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
						double    *liVec =   lftMat + lmkiIdx * dimNum;
						double    *biVec =  btmiMat + lmkiIdx * dimNum;
						double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
						double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
							double    *rjVec =   rgtMat + lmkjIdx * dimNum;
							double    *bjVec =  btmjMat + lmkjIdx * dimNum;
							double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
							double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

							double biqVal = 0.0, bjqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
								bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double d1qKVal = -(4.0 * p0Val + (8.0 + dijSqu) * p1Val);
							double d2qKVal = p0Val + 2.0 * p1Val;
							double  lrsVal = dotProduct(liVec, rjVec, dimNum) / (48.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     d1qKVal * biVec[dimIdx]
								   );

								dqijKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     -d2qKVal * biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     d1qKVal * biVec[dimIdx]
								   );

								dqjiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     -d2qKVal * bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  -
								     d1qKVal * bjVec[dimIdx]
								   );

								dqjjKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  +
								     d1qKVal * bjVec[dimIdx]
									);
							}
						}
					}
					break;

				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
						double    *liVec =   lftMat + lmkiIdx * dimNum;
						double    *biVec =  btmiMat + lmkiIdx * dimNum;
						double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
						double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
							double    *rjVec =   rgtMat + lmkjIdx * dimNum;
							double    *bjVec =  btmjMat + lmkjIdx * dimNum;
							double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
							double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

							double biqVal = 0.0, bjqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
								bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double dijSqu = dijVal * dijVal;

							double p0Val, p1Val;
							p0Fcn(p0Val, dijVal);
							p1Fcn(p1Val, dijVal);

							double d1qKVal = -((24.0 + dijSqu) * p0Val + 8.0 * (6.0 + dijSqu) * p1Val);
							double d2qKVal = 4.0 * p0Val + (8.0 + dijSqu) * p1Val;
							double  lrsVal = dotProduct(liVec, rjVec, dimNum) / (384.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     d1qKVal * biVec[dimIdx]
								   );

								dqijKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     -d2qKVal * biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     d1qKVal * biVec[dimIdx]
								   );

								dqjiKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     -d2qKVal * bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  -
								     d1qKVal * bjVec[dimIdx]
								   );

								dqjjKVec[dimIdx] += 
								   lrsVal
								   *
								   (
								     d2qKVal * bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  +
								     d1qKVal * bjVec[dimIdx]
									);
							}
						}
					}
					break;
			}
		}
		else   // dimNum == 3
		{
			switch ( knlOrder )
			{
				case 2:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
						double    *liVec =   lftMat + lmkiIdx * dimNum;
						double    *biVec =  btmiMat + lmkiIdx * dimNum;
						double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
						double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
							double    *rjVec =   rgtMat + lmkjIdx * dimNum;
							double    *bjVec =  btmjMat + lmkjIdx * dimNum;
							double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
							double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

							double biqVal = 0.0, bjqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
								bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double lreVal = dotProduct(liVec, rjVec, dimNum) * exp(-dijVal) / (3.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     (1.0 + dijVal) * biVec[dimIdx]
								   );

								dqijKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     -biqVal / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     (1.0 + dijVal) * biVec[dimIdx]
								   );

								dqjiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     -bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  +
								     (1.0 + dijVal) * bjVec[dimIdx]
								   );

								dqjjKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     bjqVal / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  -
								     (1.0 + dijVal) * bjVec[dimIdx]
								   );
							}
						}
					}
					break;
			
				case 3:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
						double    *liVec =   lftMat + lmkiIdx * dimNum;
						double    *biVec =  btmiMat + lmkiIdx * dimNum;
						double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
						double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
							double    *rjVec =   rgtMat + lmkjIdx * dimNum;
							double    *bjVec =  btmjMat + lmkjIdx * dimNum;
							double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
							double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

							double biqVal = 0.0, bjqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
								bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double lreVal = dotProduct(liVec, rjVec, dimNum) * exp(-dijVal) / (15.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     biqVal * (1.0 + dijVal) / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     (3.0 + dijVal * (3.0 + dijVal)) * biVec[dimIdx]
								   );

								dqijKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     -biqVal * (1.0 + dijVal) / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     (3.0 + dijVal * (3.0 + dijVal)) * biVec[dimIdx]
								   );

								dqjiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     -bjqVal * (1.0 + dijVal) / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  +
								     (3.0 + dijVal * (3.0 + dijVal)) * bjVec[dimIdx]
								   );

								dqjjKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     bjqVal * (1.0 + dijVal) / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  -
								     (3.0 + dijVal * (3.0 + dijVal)) * bjVec[dimIdx]
								   );
							}
						}
					}
					break;
			
				case 4:
					for ( int lmkiIdx = 0; lmkiIdx < lmkiNum; ++lmkiIdx )
					{
						double    *qiVec =  lmkiMat + lmkiIdx * dimNum;
						double    *liVec =   lftMat + lmkiIdx * dimNum;
						double    *biVec =  btmiMat + lmkiIdx * dimNum;
						double *dqiiKVec = dqiiKMat + lmkiIdx * dimNum;
						double *dqjiKVec = dqjiKMat + lmkiIdx * dimNum;

						for ( int lmkjIdx = 0; lmkjIdx < lmkjNum; ++lmkjIdx )
						{
							double    *qjVec =  lmkjMat + lmkjIdx * dimNum;
							double    *rjVec =   rgtMat + lmkjIdx * dimNum;
							double    *bjVec =  btmjMat + lmkjIdx * dimNum;
							double *dqijKVec = dqijKMat + lmkjIdx * dimNum;
							double *dqjjKVec = dqjjKMat + lmkjIdx * dimNum;

							double biqVal = 0.0, bjqVal = 0.0;
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								biqVal += biVec[dimIdx] * (qiVec[dimIdx] - qjVec[dimIdx]);
								bjqVal += bjVec[dimIdx] * (qjVec[dimIdx] - qiVec[dimIdx]);
							}

							double dijVal = eucdist(qiVec, qjVec, dimNum) / knlWidth;
							double lreVal = dotProduct(liVec, rjVec, dimNum) * exp(-dijVal) / (105.0 * knlWidthSqu);
							
							for ( int dimIdx = 0; dimIdx < dimNum; ++dimIdx )
							{
								dqiiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     biqVal * (3.0 + dijVal * (3.0 + dijVal)) / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  -
								     (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * biVec[dimIdx]
								   );

								dqijKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     -biqVal * (3.0 + dijVal * (3.0 + dijVal)) / knlWidthSqu * (qiVec[dimIdx] - qjVec[dimIdx])
									  +
								     (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * biVec[dimIdx]
								   );

								dqjiKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     -bjqVal * (3.0 + dijVal * (3.0 + dijVal)) / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  +
								     (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * bjVec[dimIdx]
								   );

								dqjjKVec[dimIdx] += 
								   lreVal
								   *
								   (
								     bjqVal * (3.0 + dijVal * (3.0 + dijVal)) / knlWidthSqu * (qjVec[dimIdx] - qiVec[dimIdx])
									  -
								     (15.0 + dijVal * (15.0 + dijVal * (6.0 + dijVal))) * bjVec[dimIdx]
								   );
							}
						}
					}
					break;
			}
		}
	}

	return;
}
