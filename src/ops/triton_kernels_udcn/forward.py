import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 32,}, num_stages=1, num_warps=1),
        # triton.Config({'BLOCK_SIZE': 64,}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64, }, num_stages=1, num_warps=1),
    ],
    key=['B', 'H', 'W', 'N', 'G', 'C', 'K'],
)
@triton.jit
def _forward_kernel(
        B: tl.constexpr,
        H: tl.constexpr, # image_size_h
        W: tl.constexpr, # image_size_w
        N: tl.constexpr,
        G: tl.constexpr, # num_channels_per_group
        C: tl.constexpr, # num_groups
        K: tl.constexpr, # kernel size
        input_ptr,   # input features [B, H, W, G, C]
        deformable_ptr, # deformable offsets [B, H, W, G, 2*K + K]
        weights_ptr, # weights [B, H, W, G, K]
        out_ptr, # out [B, H, W, G, C]
        BLOCK_SIZE: tl.constexpr, # a micro block to process in the Group
):
    pid = tl.program_id(0)
    # wid = pid % W
    # hid = pid // W % H
    nid = pid % N
    gid = pid // (N) % G
    bid = pid // (N * G)

    id_mask = (nid < N) & (gid < G) & (bid < B)
    common_offset = bid*N*G + nid*G + gid
    batch_base = bid * H * W * G * C

    for block_base in tl.static_range(0, C, BLOCK_SIZE):
        buffer = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
        block_offset = tl.arange(0, BLOCK_SIZE) + block_base
        block_mask = (block_offset < C) & id_mask
        for k in tl.static_range(K):
            deformable_offset = (common_offset * K + k) * 2
            x = tl.load(deformable_ptr + deformable_offset, mask=id_mask, other=0.0)
            y = tl.load(deformable_ptr + deformable_offset + 1, mask=id_mask, other=0.0)

            floor_x = x.to(tl.int32)
            floor_y = y.to(tl.int32)
            ceil_x = floor_x + 1
            ceil_y = floor_y + 1

            # load top left
            tl_weight = (ceil_x - x) * (ceil_y - y)
            tl_block_offset = (batch_base + floor_y * W * G * C + floor_x * G * C + gid * C) #+ k * BLOCK_SIZE
            tl_block_mask = (floor_y >= 0) & (floor_x >= 0) & (floor_x < W) & (floor_y < H)

            # load top right
            tr_weight = (x - floor_x) * (ceil_y - y)
            tr_block_offset = (batch_base + floor_y * W * G * C + ceil_x * G * C + gid * C) #+ k * BLOCK_SIZE
            tr_block_mask = (floor_y >= 0) & (ceil_x < W) & (floor_y < H) & (ceil_x >= 0)
            # load bottom left
            bl_weight = (ceil_x - x) * (y - floor_y)
            bl_block_offset = (batch_base + ceil_y * W * G * C + floor_x * G * C + gid * C) #+ k * BLOCK_SIZE
            bl_block_mask = (ceil_y < H) & (ceil_y >= 0) & (floor_x < W) & (floor_x >= 0)
            # load bottom right
            br_weight = (x - floor_x) * (y - floor_y)
            br_block_offset = (batch_base + ceil_y * W * G * C + ceil_x * G * C + gid * C) #+ k * BLOCK_SIZE
            br_block_mask = (ceil_y < H) & (ceil_y >= 0) & (ceil_x < W) & (ceil_x >= 0)

            # load dynamic weight and mask
            weights_offset = common_offset*K + k
            weight = tl.load(weights_ptr + weights_offset, mask=id_mask, other=0.0)



            tl_block_input = tl.load(input_ptr + tl_block_offset + block_offset, mask=tl_block_mask & block_mask, other=0.0)
            tl_block_input = tl_block_input * tl_weight

            # load top right
            tr_block_input = tl.load(input_ptr + tr_block_offset + block_offset, mask=tr_block_mask & block_mask, other=0.0)
            tr_block_input = tr_block_input * tr_weight
            # load bottom left
            bl_block_input = tl.load(input_ptr + bl_block_offset + block_offset, mask=bl_block_mask & block_mask, other=0.0)
            bl_block_input = bl_block_input * bl_weight
            # load bottom right
            br_block_input = tl.load(input_ptr + br_block_offset + block_offset, mask=br_block_mask & block_mask, other=0.0)
            br_block_input = br_block_input * br_weight

            # sampled
            sampled_input = tl_block_input + tr_block_input + bl_block_input + br_block_input

            weighted_sampled_input = sampled_input * weight
            buffer = buffer + weighted_sampled_input
        # store to out_ptr
        tl.store(out_ptr + common_offset*C + block_offset, buffer, mask=block_mask)

