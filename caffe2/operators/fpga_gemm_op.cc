#include "caffe2/operators/fpga_gemm_op.h"
#include "caffe2/fpga/xcl2.hpp"
//need to ask how fpga works, is this header file enough?

namespace caffe2 {
namespace {
//gemm is actually im2col and matmul and col2im?
//here is to implement these three tasks on fpga?
REGISTER_FPGA_OPERATOR(FPGAGEMM, FPGAGEMMOp<float, FPGAContext>);


}//namespace
}//namespace caffe2
