mexcuda -DDIM2 -I.. computeKernelITF.cu computeKernel.cu -output computeKernel
mexcuda -DDIM2 -I.. multiplyKernelITF.cu multiplyKernel.cu -output multiplyKernel
mexcuda -DDIM2 -I.. dqKernelITF.cu dqKernel.cu -output dqKernel
mexcuda -DDIM2 -I.. d2qKernelITF.cu d2qKernel.cu -output d2qKernel
