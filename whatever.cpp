//Implements gemm for FPGA
#include "caffe2/utils/math.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "caffe2/core/context.h"
//#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/fixed_divisor.h"

#include "Eigen/Core"
#include "Eigen/Dense"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

// #ifdef CAFFE2_USE_HPTT
// #include <hptt.h>
// #endif // CAFFE2_USE_HPTT

#ifdef CAFFE2_USE_XCL
#include "caffe2/fpga/xcl2.hpp"
#endif // CAFFE2_USE_XCL

//the below are from math_functions.cpp
//this form is from the std library so it's fine?
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <limits>
#include <fstream>

//these are from caffe, replacement in caffe2?
//#include "caffe/common.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/util/rng.hpp"

//this is for timer
//from https://github.com/pytorch/pytorch/blob/master/caffe2/core/timer.h
#include "caffe2/core/timer.h"
// #if defined(_MSC_VER)
// #include <process.h>
// #endif

namespace caffe2 {
namespace math {
//fpga kernel only needs support form xcl library?
//keep MKL to call cblas for verification

#ifdef CAFFE2_USE_XCL_FOR_BLAS
template <>
C10_EXPORT void Gemm<float, FPGAContext>(
    const CBLAS_TRANSPOSE trans_A,
    const CBLAS_TRANSPOSE trans_B,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float* A,
    const float* B,
    const float beta,
    float* C,
    FPGAContext* context,
    TensorProto::DataType math_type) {
      //from caffe_fpga math_functions.cpp
      #ifdef PROFILING
      std::ofstream profilingLog;
      profilingLog.open("/home/centos/src/project_data/caffe/huawei_proj/profile/log.csv", ios::app);

      profilingLog << TILE_ROW << "," << TILE_COL << "," << TILE_COMMON << "," << M << "," << N << "," << K << ",";

      //changed in caffe2 math_cpu.cc to the below form, the min would be 1, need to change?
      //about that -111 so it becomes either 1/0
      // int lda = (TransA == CblasNoTrans) ? K : M;
      // int ldb = (TransB == CblasNoTrans) ? N : K;
      const int lda = std::max((trans_A == CblasNoTrans) ? K : M, 1);
      const int ldb = std::max((trans_B == CblasNoTrans) ? N : K, 1);

      //caffe::Timer cpu;
      caffe2::Timer cpu;
      double cpu_time = 0.0;
      cpu.Start();

      //for verification
      cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);

      cpu_time += cpu.MicroSeconds();
      profilingLog << cpu_time / 1000.0 << ",";

      float cBlas[N*M];
      for (int i=0; i<M; i++)
      {
          for (int j=0; j<N; j++)
          {
              cBlas[i*N + j] = C[i*N+j];
          }
      }
      #endif

      double fpga_times[4];

      #ifdef SIMULATE_BATCHING
      Kernel_profiling(TransA - 111, A, TransB - 111, B, C, M, K, K, N, alpha, beta);
      exit(EXIT_SUCCESS);
      #else

      Kernel(TransA - 111, A, TransB - 111, B, C, M, K, K, N, alpha, beta, fpga_times);
      // Kernel_double_buff(TransA - 111, A, TransB - 111, B, C, M, K, K, N, alpha, beta, fpga_times);
      // Kernel_tiling(TransA - 111, A, TransB - 111, B, C, M, K, K, N, alpha, beta, fpga_times);
      // Kernel_double_ddr(TransA - 111, A, TransB - 111, B, C, M, K, K, N, alpha, beta, fpga_times);

      #endif

      #ifdef PROFILING

      #ifdef PROFILING_TIME
      double total_time = 0.0;
      for (unsigned i=0; i<4; i++)
      {
          total_time += fpga_times[i];
      }
      profilingLog << fpga_times[0] << "," << fpga_times[1] << "," << fpga_times[2] << "," << fpga_times[3] << "," << total_time << ",";
      #endif

      double mse = 0;
      for (int i=0; i<M; i++)
      {
          for (int j=0; j<N; j++)
          {
      	    mse += std::pow(std::fabs(cBlas[i*N+j] - C[i*N+j]) ,2);
          }
      }
      mse /= (N*M);

      profilingLog << mse << std::endl;
      // LOG(INFO) << "MSE = " << mse;

      profilingLog.close();
      #endif
}
}//math
}//caffe2
