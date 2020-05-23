/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#ifndef _DEFINES_H_
#define _DEFINES_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include <string>
#include <chrono>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#if GPU == 1
#include <cublas_v2.h>
#include <curand.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#if STOCHASTIC == 1 && FIXEDPOINT == 1
#error "Can not set STOCHASTIC and FIXEDPOINT at the same time"
#elif STOCHASTIC == 1
#include "stochastic.h"
typedef SC<64, 2048> DataType;
#elif FIXEDPOINT == 1
#include "fixed_point.h"
typedef FixedPoint<int, 22, 10> DataType;
#else
typedef float DataType;
#endif

///////////////////////////////////////////////
///////////////////////////////////////////////
///////////////////////////////////////////////

using namespace std;

extern bool use_gpu;

#define OPENMP_THREADS 8
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#define EPSILON 1e-4
#if GPU == 1
#define BLOCK 512
#endif

class MemoryMoniter;
template<typename T>
class Tensor;
class Params;
class Layer;
class LayerInput;
class LayerAct;
class LayerConv;
class LayerFull;
class LayerPool;
class LayerBN;
class LayerDrop;
class Net;

/* assert */
inline void Assert(bool b, std::string msg)
{
    if (b)
    {
        return;
    }
    std::string _errmsg = std::string("Assertion Failed: ") + msg;
    std::cerr << _errmsg.c_str() << std::endl;
    exit(1);
}

#if GPU == 1
#include "cuda_util.h"
#endif
#include "args.h"
#include "gemm.h"
#include "blas.h"
#include "conv.h"
#include "MemoryMonitor.h"
#include "Tensor.h"
#include "Params.h"
#include "Layer.h"
#include "LayerInput.h"
#include "LayerAct.h"
#include "LayerConv.h"
#include "LayerFull.h"
#include "LayerPool.h"
#include "LayerBN.h"
#include "LayerDrop.h"
#include "Net.h"

#endif
