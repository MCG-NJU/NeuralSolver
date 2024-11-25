#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <iostream>
#include <vector_types.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>
#include <cuda_bf16.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>


namespace cg = cooperative_groups;

template<typename scalar_t>
__device__ __always_inline int toInt(scalar_t val);

template<>
__device__  __always_inline int toInt(float val){
    return static_cast<int>(val);
}
template<>
__device__  __always_inline int toInt(half val){
    return __half2int_rz(val);
}

template<typename scalar_t>
__device__ __always_inline scalar_t fromInt(int val);

template<>
__device__ __always_inline float fromInt(int val){
    return static_cast<float>(val);
}

template<>
__device__ __always_inline half fromInt(int val){
    return __int2half_rz(val);
}

template<typename scalar_t>
__device__ __always_inline scalar_t constVal(float val);

template<>
__device__ __always_inline float constVal<float>(float val) {
    return (float)val;
}

template<>
__device__ __always_inline half constVal<half>(float val) {
    return __float2half(val); // Using float to half conversion
}
template<>
__device__ __always_inline nv_bfloat16 constVal<nv_bfloat16>(float val){
    return __float2bfloat16(val);
}





// B, H, W, C, BLOCK_DIM must be multiple of C
template <typename scalar_t, typename vec2_t, int pipeline_stages, int TILE_C, int TILE_THREADS>
__global__ void dcn_backward_pipeline_kernel(
        const int H,
        const int W,
        const int G,
        const int K,
        const int C,
        scalar_t* ptr_values,
        scalar_t* ptr_deformables,
        scalar_t* ptr_weights,
        scalar_t* ptr_grad_out,
        scalar_t* ptr_grad_values,
        scalar_t* ptr_grad_deformables,
        scalar_t* ptr_grad_weights
) {
    auto block = cg::this_thread_block();
    auto self_thread = cg::this_thread();
    auto tile_threads = cg::tiled_partition<TILE_THREADS>(block);
    int local_thread_id = block.thread_rank();
    int local_tile_id = tile_threads.meta_group_rank();
    int num_local_tiles = tile_threads.meta_group_size();
    int global_tile_id = block.group_index().x*num_local_tiles + local_tile_id;

    extern __shared__ int shm[];
    auto GradBuffer = reinterpret_cast<scalar_t*>(shm);
    scalar_t* Buffer = reinterpret_cast<scalar_t*>(shm) + num_local_tiles*C;
    if(global_tile_id >= H*W*G) return;

    int bid = block.group_index().y;
    int gid = global_tile_id % G;
    int wid = global_tile_id / G % W;
    int hid = global_tile_id / G / W;
    int globale_offset = bid*H*W*G*C + global_tile_id*C;
    cg::memcpy_async(tile_threads, GradBuffer+local_tile_id*C, ptr_grad_out+globale_offset, sizeof(scalar_t)*C);

    int shared_offset[pipeline_stages];
    for (int s = 0; s < pipeline_stages; ++s) {
        shared_offset[s] = (s+pipeline_stages*local_thread_id)*(TILE_C*4);
    }

    auto pipeline = cuda::make_pipeline();
    const int num_tiles_per_thread = C/TILE_C/TILE_THREADS;

    for(int k=0; k<K; k++) {
        int offset = bid * K * H * W * G + hid * W * K * G + wid * K * G + gid * K + k;
        scalar_t x, y, weight;
        if (tile_threads.thread_rank() == 0) {
            x = ptr_deformables[offset*2] + fromInt<scalar_t>(wid);
            y = ptr_deformables[offset*2 + 1] + fromInt<scalar_t>(hid);
//            x = fromInt<scalar_t>(wid);
//            y = fromInt<scalar_t>(hid);
            weight = ptr_weights[offset];
        }
        tile_threads.sync();
        x = tile_threads.shfl(x, 0);
        y = tile_threads.shfl(y, 0);
        weight = tile_threads.shfl(weight, 0);

        int floor_x = toInt<scalar_t>(x);
        int floor_y = toInt<scalar_t>(y);
        int ceil_x = floor_x + 1;
        int ceil_y = floor_y + 1;


        scalar_t dodx = constVal<scalar_t>(0.0f);
        scalar_t dody = constVal<scalar_t>(0.0f);
        scalar_t dodw = constVal<scalar_t>(0.0f);

        int start_c = tile_threads.thread_rank() * (C / TILE_THREADS);

        bool tl_flag = (floor_x >=0) and (floor_x <W) and (floor_y>=0) and (floor_y<H);
        bool tr_flag = (ceil_x >=0) and (ceil_x <W) and (floor_y>=0) and (floor_y<H);
        bool bl_flag = (floor_x >=0) and (floor_x <W) and (ceil_y>=0) and (ceil_y<H);
        bool br_flag = (ceil_x >=0) and (ceil_x <W) and (ceil_y>=0) and (ceil_y<H);

        int tl_global_base = (bid * H * W * G + floor_y * W * G + floor_x * G + gid)*C + start_c;
        int tr_global_base = (bid * H * W * G + floor_y * W * G + ceil_x * G + gid)*C  + start_c;
        int bl_global_base = (bid * H * W * G + ceil_y * W * G + floor_x * G + gid)*C +start_c;
        int br_global_base = (bid * H * W * G + ceil_y * W * G + ceil_x * G + gid)*C  +start_c;


        auto asmem_load_fn = [&](int shm_offset, int hbm_offset, bool flag){
            if(flag){
                cuda::memcpy_async(Buffer + shm_offset, ptr_values + hbm_offset,
                                   TILE_C * sizeof(scalar_t), pipeline);
            }else{
                memset(Buffer+shm_offset, TILE_C, sizeof(scalar_t));
            }
        };

        // pipeline-compute&load
        for (int compute_n = 0, fetch_n=0; compute_n < num_tiles_per_thread; compute_n++) {
            for (; fetch_n < compute_n + pipeline_stages and fetch_n < num_tiles_per_thread; fetch_n++) {
                pipeline.producer_acquire();
                int buffer_offset = shared_offset[fetch_n % pipeline_stages];

                // tl
                asmem_load_fn(buffer_offset, tl_global_base + fetch_n * TILE_C, tl_flag);
                // tr
                asmem_load_fn(buffer_offset+TILE_C, tr_global_base + fetch_n * TILE_C, tr_flag);
                // bl
                asmem_load_fn(buffer_offset+TILE_C*2, bl_global_base + fetch_n * TILE_C, bl_flag);
                // br
                asmem_load_fn(buffer_offset+TILE_C*3, br_global_base + fetch_n * TILE_C, br_flag);

                pipeline.producer_commit();
            }
            pipeline.consumer_wait();
            int buffer_id = compute_n % pipeline_stages;
            int ibuffer_offset = shared_offset[buffer_id];
            int gbuffer_offset = local_tile_id * C + start_c + compute_n * TILE_C;

            for (int j = 0; j < TILE_C; j+=2) {
                if(tl_flag){
                    // tl
                    dodw = dodw + (fromInt<scalar_t>(ceil_x) - x) * (fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+j] * GradBuffer[gbuffer_offset + j];
                    dodx = dodx + -weight*(fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+j] * GradBuffer[gbuffer_offset + j];
                    dody = dody + -weight*(fromInt<scalar_t>(ceil_x) - x) * Buffer[ibuffer_offset+j] * GradBuffer[gbuffer_offset + j];
                    dodw = dodw + (fromInt<scalar_t>(ceil_x) - x) * (fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+j + 1] * GradBuffer[gbuffer_offset + j + 1];
                    dodx = dodx + -weight*(fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+j+ 1] * GradBuffer[gbuffer_offset + j + 1];
                    dody = dody + -weight*(fromInt<scalar_t>(ceil_x) - x) * Buffer[ibuffer_offset+j + 1] * GradBuffer[gbuffer_offset + j + 1];
                    {
                        vec2_t vtl_di;
                        vtl_di.x = weight* (fromInt<scalar_t>(ceil_x) - x) * (fromInt<scalar_t>(ceil_y) - y) * GradBuffer[gbuffer_offset + j];
                        vtl_di.y = weight* (fromInt<scalar_t>(ceil_x) - x) * (fromInt<scalar_t>(ceil_y) - y) * GradBuffer[gbuffer_offset + j + 1];
                        atomicAdd((vec2_t*)(ptr_grad_values + tl_global_base + compute_n * TILE_C + j), vtl_di);
                    }
                }


                if(tr_flag){
                    // tr
                    dodw = dodw + (x - fromInt<scalar_t>(floor_x)) * (fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+TILE_C+j] * GradBuffer[gbuffer_offset + j];
                    dodx = dodx + weight*(fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+TILE_C+j] * GradBuffer[gbuffer_offset + j];
                    dody = dody + -weight*(x - fromInt<scalar_t>(floor_x)) * Buffer[ibuffer_offset+TILE_C+j] * GradBuffer[gbuffer_offset + j];
                    dodw = dodw + (x - fromInt<scalar_t>(floor_x)) * (fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+TILE_C+j+1] * GradBuffer[gbuffer_offset + j+1];
                    dodx = dodx + weight*(fromInt<scalar_t>(ceil_y) - y) * Buffer[ibuffer_offset+TILE_C+j + 1] * GradBuffer[gbuffer_offset + j+ 1];
                    dody = dody + -weight*(x - fromInt<scalar_t>(floor_x)) * Buffer[ibuffer_offset+TILE_C+j + 1] * GradBuffer[gbuffer_offset + j+1];
                    {
                        vec2_t vtr_di;
                        vtr_di.x = weight* (x - fromInt<scalar_t>(floor_x)) * (fromInt<scalar_t>(ceil_y) - y) * GradBuffer[gbuffer_offset + j];
                        vtr_di.y = weight* (x - fromInt<scalar_t>(floor_x)) * (fromInt<scalar_t>(ceil_y) - y) * GradBuffer[gbuffer_offset + j+1];
                        atomicAdd((vec2_t*)(ptr_grad_values + tr_global_base + compute_n * TILE_C + j), vtr_di);
                    }
                }

                if(bl_flag){
                    // bl
                    dodw = dodw + (fromInt<scalar_t>(ceil_x) - x) * (y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*2+j] * GradBuffer[gbuffer_offset + j];
                    dodx = dodx + -weight*(y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*2+j] * GradBuffer[gbuffer_offset + j];
                    dody = dody + weight*(fromInt<scalar_t>(ceil_x) - x) * Buffer[ibuffer_offset+TILE_C*2+j] * GradBuffer[gbuffer_offset + j];
                    dodw = dodw + (fromInt<scalar_t>(ceil_x) - x) * (y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*2+j+1] * GradBuffer[gbuffer_offset + j+1];
                    dodx = dodx + -weight*(y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*2+j+1] * GradBuffer[gbuffer_offset + j+1];
                    dody = dody + weight*(fromInt<scalar_t>(ceil_x) - x) * Buffer[ibuffer_offset+TILE_C*2+j+1] * GradBuffer[gbuffer_offset + j+1];
                    {
                        vec2_t vbl_di;
                        vbl_di.x = weight* (fromInt<scalar_t>(ceil_x) - x) * (y - fromInt<scalar_t>(floor_y)) * GradBuffer[gbuffer_offset + j];
                        vbl_di.y = weight* (fromInt<scalar_t>(ceil_x) - x) * (y - fromInt<scalar_t>(floor_y)) * GradBuffer[gbuffer_offset + j+1];
                        atomicAdd((vec2_t*)(ptr_grad_values + bl_global_base + compute_n * TILE_C + j), vbl_di);
                    }
                }


                if(br_flag){
                    // tr
                    dodw = dodw + (x - fromInt<scalar_t>(floor_x)) * (y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*3+j] * GradBuffer[gbuffer_offset + j];
                    dodx = dodx + weight*(y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*3+j] * GradBuffer[gbuffer_offset + j];
                    dody = dody + weight*(x - fromInt<scalar_t>(floor_x)) * Buffer[ibuffer_offset+TILE_C*3+j] * GradBuffer[gbuffer_offset + j];
                    dodw = dodw + (x - fromInt<scalar_t>(floor_x)) * (y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*3+j+1] * GradBuffer[gbuffer_offset + j+1];
                    dodx = dodx + weight*(y - fromInt<scalar_t>(floor_y)) * Buffer[ibuffer_offset+TILE_C*3+j+1] * GradBuffer[gbuffer_offset + j+1];
                    dody = dody + weight*(x - fromInt<scalar_t>(floor_x)) * Buffer[ibuffer_offset+TILE_C*3+j+1] * GradBuffer[gbuffer_offset + j+1];
                    {
                        vec2_t vbr_di;
                        vbr_di.x = weight* (x - fromInt<scalar_t>(floor_x)) * (y - fromInt<scalar_t>(floor_y)) * GradBuffer[gbuffer_offset + j];
                        vbr_di.y = weight* (x - fromInt<scalar_t>(floor_x)) * (y - fromInt<scalar_t>(floor_y)) * GradBuffer[gbuffer_offset + j+1];
                        atomicAdd((vec2_t*)(ptr_grad_values + br_global_base + compute_n * TILE_C + j), vbr_di);
                    }
                }
            }
            pipeline.consumer_release();
        }
        for (int i = TILE_THREADS>>1; i > 0; i/=2) {
            dodx = dodx + tile_threads.shfl_down(dodx, i);
            dody = dody + tile_threads.shfl_down(dody, i);
            dodw = dodw + tile_threads.shfl_down(dodw, i);
        }
        if (tile_threads.thread_rank() == 0) {
            cuda::memcpy_async(ptr_grad_deformables + offset * 2, &dodx, sizeof(scalar_t), pipeline);
            cuda::memcpy_async(ptr_grad_deformables + offset * 2 + 1, &dody, sizeof(scalar_t), pipeline);
            cuda::memcpy_async(ptr_grad_weights + offset, &dodw, sizeof(scalar_t), pipeline);
        }
    }
}


using namespace torch;
template<int pipeline_stages, int TILE_C, int TILE_THREADS, int THREADS>
void backward(const int B,
              const int H,
              const int W,
              const int G,
              const int K,
              const int C,
              torch::Tensor values,
              torch::Tensor deformables,
              torch::Tensor weights,
              torch::Tensor grad_out,
              torch::Tensor grad_values,
              torch::Tensor grad_deformables,
              torch::Tensor grad_weights
) {
    int num_local_tiles =(THREADS/TILE_THREADS);
    int num_global_tiles = (H*W*G+num_local_tiles-1)/num_local_tiles;
    dim3 launch_threads_per_block(THREADS);
    dim3 launch_blocks(num_global_tiles, B);

    int deformable_shm_size = 0;
    int grad_out_shm_size = num_local_tiles*C;
    int pipeline_shm_size = (pipeline_stages*TILE_C*4*THREADS);

    int shm_size = deformable_shm_size+grad_out_shm_size+pipeline_shm_size;
//    printf("shm_size: %d\n", shm_size/512);
//    printf("pipeline_size: %d\n", pipeline_shm_size/512);
//    printf("grad_out_size: %d\n", grad_out_shm_size/512);


    switch (values.type().scalarType()) {
        case at::ScalarType::Half:
            return dcn_backward_pipeline_kernel<half, half2, pipeline_stages, TILE_C, TILE_THREADS><<<launch_blocks, launch_threads_per_block, shm_size*sizeof(half)>>>(
                    H, W, G, K, C,
                    reinterpret_cast<half*>(values.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(deformables.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(weights.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(grad_out.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(grad_values.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(grad_deformables.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(grad_weights.data_ptr<at::Half>())
            );
//        case at::ScalarType::BFloat16:
//            return dcn_backward_pipeline_kernel<nv_bfloat16, nv_bfloat162, pipeline_stages, TILE_C, TILE_THREADS><<<launch_blocks, launch_threads_per_block, shm_size*sizeof(nv_bfloat16)>>>(
//                    H, W, G, K, C,
//                    reinterpret_cast<nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
//                    reinterpret_cast<nv_bfloat16*>(deformables.data_ptr<at::BFloat16>()),
//                    reinterpret_cast<nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
//                    reinterpret_cast<nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
//                    reinterpret_cast<nv_bfloat16*>(grad_values.data_ptr<at::BFloat16>()),
//                    reinterpret_cast<nv_bfloat16*>(grad_deformables.data_ptr<at::BFloat16>()),
//                    reinterpret_cast<nv_bfloat16*>(grad_weights.data_ptr<at::BFloat16>())
//            );
        default:
            printf("running error");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_p1_c2_tile16_thread128", &backward<1, 2, 16, 128>, "");
    m.def("backward_p2_c2_tile16_thread128", &backward<2, 2, 16, 128>, "");
    m.def("backward_p1_c4_tile16_thread128", &backward<1, 4, 16, 128>, "");
    m.def("backward_p1_c2_tile16_thread256", &backward<1, 2, 16, 256>, "");
    m.def("backward_p2_c2_tile16_thread256", &backward<2, 2, 16, 256>, "");
    m.def("backward_p1_c4_tile16_thread256", &backward<1, 4, 16, 256>, "");
    m.def("backward_p1_c2_tile16_thread384", &backward<1, 2, 16, 384>, "");
    m.def("backward_p2_c2_tile16_thread384", &backward<2, 2, 16, 384>, "");
    m.def("backward_p1_c4_tile16_thread384", &backward<1, 4, 16, 384>, "");
    m.def("backward_p1_c2_tile16_thread512", &backward<1, 2, 16, 512>, "");
    m.def("backward_p2_c2_tile16_thread512", &backward<2, 2, 16, 512>, "");
    m.def("backward_p1_c4_tile16_thread512", &backward<1, 4, 16, 512>, "");
    m.def("backward_p1_c2_tile16_thread768", &backward<1, 2, 16, 768>, "");
    m.def("backward_p2_c2_tile16_thread768", &backward<2, 2, 16, 768>, "");
    m.def("backward_p1_c4_tile16_thread768", &backward<1, 4, 16, 768>, "");
//    m.def("backward_p1_c2_tile16_thread1024", &backward<1, 2, 16, 1024>, "");
//    m.def("backward_p2_c2_tile16_thread1024", &backward<2, 2, 16, 1024>, "");
//    m.def("backward_p1_c4_tile16_thread1024", &backward<1, 4, 16, 1024>, "");

    m.def("backward_p1_c2_tile32_thread128", &backward<1, 2, 32, 128>, "");
    m.def("backward_p1_c2_tile32_thread256", &backward<1, 2, 32, 256>, "");
    m.def("backward_p1_c2_tile32_thread384", &backward<1, 2, 32, 384>, "");
    m.def("backward_p1_c2_tile32_thread512", &backward<1, 2, 32, 512>, "");
}
