nvcc -o gemm_fp16 gemm_fp16_acc_fp16_test.cu -lcublas

./gemm_fp16
