% Compile the files. This only needs to be run once.
compile

%%

% --------------- %
%  computeKernel  %
% --------------- %

dimNum = 2;     % number of dimensions, can be any positive integer
pntNum = 100;   % number of points
datMat = rand(dimNum, pntNum) - 0.5;   % datMat = [q1, q2, ..., qn]

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

knlMat = computeKernel(datMat, knlOrder, knlWidth);   % knlMat(i, j) = k(qi, qj)

%%

dimNum  = 2;    % number of dimensions, can be any positive integer
pnt1Num = 100;  % number of points
pnt2Num = 200;  % number of points
dat1Mat = rand(dimNum, pnt1Num) - 0.5;   % dat1Mat = [r1, r2, ..., rm]
dat2Mat = rand(dimNum, pnt2Num) - 0.5;   % dat2Mat = [s1, s2, ..., sn]

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

knlMat = computeKernel(dat1Mat, dat2Mat, knlOrder, knlWidth);   % knlMat(i, j) = k(ri, sj)

%%

% ---------------- %
%  multiplyKernel  %
% ---------------- %

dimNum = 2;     % number of dimensions, can be any positive integer
pntNum = 100;   % number of points
datMat = rand(dimNum, pntNum) - 0.5;   % datMat = [q1, q2, ..., qn]
alpMat = rand(dimNum, pntNum) - 0.5;   % alpMat = [a1, a2, ..., an]

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

vlcMat = multiplyKernel(datMat, alpMat, knlOrder, knlWidth);   % vlcMat(:, i) = sum_j k(qi, qj) aj

%%

dimNum  = 2;    % number of dimensions, can be any positive integer
pnt1Num = 100;  % number of points
pnt2Num = 200;  % number of points
dat1Mat = rand(dimNum, pnt1Num) - 0.5;   % dat1Mat = [r1, r2, ..., rm]
dat2Mat = rand(dimNum, pnt2Num) - 0.5;   % dat2Mat = [s1, s2, ..., sn]
alp2Mat = rand(dimNum, pnt2Num) - 0.5;   % alp2Mat = [a1, a2, ..., an]

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

vlcMat = multiplyKernel(dat1Mat, dat2Mat, alp2Mat, knlOrder, knlWidth);   % vlcMat(:, i) = sum_j k(qi, qj) aj

%%

% ---------- %
%  dqKernel  %
% ---------- %

dimNum = 2;     % number of dimensions, can be any positive integer
pntNum = 100;   % number of points
datMat = rand(dimNum, pntNum) - 0.5;   % datMat = [q1, q2, ..., qn]
btaMat = rand(dimNum, pntNum) - 0.5;   % btaMat = [b1, b2, ..., bn]
alpMat = rand(dimNum, pntNum) - 0.5;   % alpMat = [a1, a2, ..., an]

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

dqKMat = dqKernel(datMat, btaMat, alpMat, knlOrder, knlWidth);
% dqKMat(:, i): first derivative of sum_ij k(qi, qj) bi^T aj with respect to qi

%%

dimNum  = 2;    % number of dimensions, can be any positive integer
pnt1Num = 100;  % number of points
pnt2Num = 200;  % number of points
dat1Mat = rand(dimNum, pnt1Num) - 0.5;   % dat1Mat = [r1, r2, ..., rm]
dat2Mat = rand(dimNum, pnt2Num) - 0.5;   % dat2Mat = [s1, s2, ..., sn]
bta1Mat = rand(dimNum, pnt1Num) - 0.5;   % bta1Mat = [b1, b2, ..., bm]
alp2Mat = rand(dimNum, pnt2Num) - 0.5;   % alp2Mat = [a1, a2, ..., an]

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

[dqK1Mat, dqK2Mat] = dqKernel(dat1Mat, dat2Mat, bta1Mat, alp2Mat, knlOrder, knlWidth);
% dqK1Mat(:, i): first derivative of sum_ij k(ri, sj) bi^T aj with respect to ri
% dqK2Mat(:, j): first derivative of sum_ij k(ri, sj) bi^T aj with respect to sj

%%

% ----------- %
%  d2qKernel  %
% ----------- %

dimNum = 2;     % number of dimensions, can be any positive integer
pntNum = 100;   % number of points
datMat = rand(dimNum, pntNum) - 0.5;   % datMat = [q1, q2, ..., qn]
btaMat = rand(dimNum, pntNum) - 0.5;
alpMat = rand(dimNum, pntNum) - 0.5;
gmaMat = rand(dimNum, pntNum) - 0.5;

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

dq2KMat = d2qKernel(datMat, btaMat, alpMat, gmaMat, knlOrder, knlWidth); 
% dq2KMat(:, i): second derivative with respect to qi

%%

dimNum  = 2;    % number of dimensions, can be any positive integer
pnt1Num = 100;  % number of points
pnt2Num = 200;  % number of points
dat1Mat = rand(dimNum, pnt1Num) - 0.5;   % dat1Mat = [r1, r2, ..., rm]
dat2Mat = rand(dimNum, pnt2Num) - 0.5;   % dat2Mat = [s1, s2, ..., sn]
bta1Mat = rand(dimNum, pnt1Num) - 0.5;
alp2Mat = rand(dimNum, pnt2Num) - 0.5;
gma1Mat = rand(dimNum, pnt1Num) - 0.5;
gma2Mat = rand(dimNum, pnt2Num) - 0.5;

knlOrder = 3;   % kernel order, -1: Gaussian kernel, 0 to 4: Matern kernel
knlWidth = 0.2; % kernel width

[dq2K1Mat, dq2K2Mat] = d2qKernel(dat1Mat, dat2Mat, bta1Mat, alp2Mat, gma1Mat, gma2Mat, knlOrder, knlWidth);
% dq2K1Mat(:, i): second derivative with respect to ri
% dq2K2Mat(:, j): second derivative with respect to sj


