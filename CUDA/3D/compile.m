mexcuda -DDIM3 -I.. computeKernelITF.cu computeKernel.cu -output computeKernel
mexcuda -DDIM3 -I.. multiplyKernelITF.cu multiplyKernel.cu -output multiplyKernel
mexcuda -DDIM3 -I.. dqKernelITF.cu dqKernel.cu -output dqKernel
mexcuda -DDIM3 -I.. d2qKernelITF.cu d2qKernel.cu -output d2qKernel
