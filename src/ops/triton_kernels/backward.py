import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64,}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64,}, num_stages=1, num_warps=2),
    ],
    key=['B', 'H', 'W', 'G', 'C', 'K'],
)
@triton.jit
def backward_kernel(
        B: tl.constexpr,
        H: tl.constexpr, # image_size_h
        W: tl.constexpr, # image_size_w
        G: tl.constexpr, # num_groups
        C: tl.constexpr, # num_channels_per_group
        K: tl.constexpr, # kernel size
        input_ptr,   # input features [B, H, W, G, C]
        deformable_ptr, # deformable offsets [B, H, W, G, K, 2]
        weights_ptr, # weights [B, H, W, G, K]
        grad_ptr, # out [B, H, W, G, C]
        grad_input_ptr,   # input features [B, H, W, G, C]
        grad_deformable_ptr, # deformable offsets [B, H, W, G, K, 2]
        grad_weights_ptr, # weights [B, H, W, G, K]
        BLOCK_SIZE: tl.constexpr, # a micro block to process in the Group
):

    pid = tl.program_id(0)
    wid = pid % W
    hid = pid // W % H
    gid = pid // (W * H) % G
    bid = pid // (W * H * G)

    id_mask = (hid < H) & (wid < W) & (gid < G) & (bid < B)

    common_offset = bid*H*W*G + hid*W*G + wid*G + gid
    batch_base = bid * H * W * G * C
    for k in tl.static_range(K):
        # load dynamic weight and mask
        weights_offset = common_offset*K + k
        weight = tl.load(weights_ptr + weights_offset, mask=id_mask, other=0.0)
        dodx = tl.zeros((1,), dtype=grad_deformable_ptr.type.element_ty)
        dody = tl.zeros((1,), dtype=grad_deformable_ptr.type.element_ty)
        dodw = tl.zeros((1,), dtype=grad_weights_ptr.type.element_ty)
        deformable_offset = (common_offset * K + k)*2
        x = tl.load(deformable_ptr + deformable_offset, mask=id_mask, other=0.0) + wid
        y = tl.load(deformable_ptr + deformable_offset + 1, mask=id_mask, other=0.0) + hid
        for block_base in tl.static_range(0, C, BLOCK_SIZE):
            block_offset = tl.arange(0, BLOCK_SIZE) + block_base
            block_mask = (block_offset < C) & id_mask
            grad = tl.load(grad_ptr+common_offset*C + block_offset, mask=block_mask, other=0.0)
            dods = weight*grad

            floor_x = x.to(tl.int32)
            floor_y = y.to(tl.int32)
            ceil_x = floor_x + 1
            ceil_y = floor_y + 1

            # load top left
            tl_weight = (ceil_x - x) * (ceil_y - y)
            tl_block_offset = (batch_base + floor_y * W * G * C + floor_x * G * C + gid * C) + block_offset
            tl_block_mask = ((floor_y >= 0) & (floor_x >= 0) & (floor_x < W) & (floor_y < H))
            tl_block_input = tl.load(input_ptr + tl_block_offset, mask=tl_block_mask & block_mask, other=0.0)
            tl_block_input_dot_grad = tl.sum(tl_block_input*grad, axis=0)
            dodx = dodx + -1 * tl_block_input_dot_grad * (ceil_y - y)
            dody = dody + -1 * tl_block_input_dot_grad * (ceil_x - x)
            dodw = dodw + tl_block_input_dot_grad * tl_weight

            dodtl = dods * tl_weight
            tl.atomic_add(grad_input_ptr + tl_block_offset, mask=tl_block_mask & block_mask, val=dodtl)


            # load top right
            tr_weight = (x - floor_x) * (ceil_y - y)
            tr_block_offset = (batch_base + floor_y * W * G * C + ceil_x * G * C + gid * C) + block_offset
            tr_block_mask = ((floor_y >= 0) & (ceil_x < W) & (floor_y < H) & (ceil_x >= 0))
            tr_block_input = tl.load(input_ptr + tr_block_offset, mask=tr_block_mask & block_mask, other=0.0)
            tr_block_input_dot_grad = tl.sum(tr_block_input*grad, axis=0)
            dodx = dodx + 1 * tr_block_input_dot_grad * (ceil_y - y)
            dody = dody + -1 * tr_block_input_dot_grad * (x - floor_x)
            dodw = dodw + tr_block_input_dot_grad*tr_weight

            dodtr = dods * tr_weight
            tl.atomic_add(grad_input_ptr + tr_block_offset, mask=tr_block_mask & block_mask, val=dodtr)


            # load bottom left
            bl_weight = (ceil_x - x) * (y - floor_y)
            bl_block_offset = (batch_base + ceil_y * W * G * C + floor_x * G * C + gid * C) + block_offset
            bl_block_mask = ((ceil_y < H) & (ceil_y >= 0) & (floor_x < W) & (floor_x >= 0))
            bl_block_input = tl.load(input_ptr + bl_block_offset, mask=bl_block_mask & block_mask, other=0.0)
            bl_block_input_dot_grad = tl.sum(bl_block_input*grad, axis=0)
            dodx = dodx + -1 * bl_block_input_dot_grad * (y - floor_y)
            dody = dody + 1 * bl_block_input_dot_grad * (ceil_x - x)
            dodw = dodw + bl_block_input_dot_grad*bl_weight

            dodbl = dods * bl_weight
            tl.atomic_add(grad_input_ptr + bl_block_offset, mask=bl_block_mask & block_mask, val=dodbl)


            # load bottom right
            br_weight = (x - floor_x) * (y - floor_y)
            br_block_offset = (batch_base + ceil_y * W * G * C + ceil_x * G * C + gid * C) + block_offset
            br_block_mask = ((ceil_y < H) & (ceil_y >= 0) & (ceil_x < W) & (ceil_x >= 0))
            br_block_input = tl.load(input_ptr + br_block_offset, mask=br_block_mask & block_mask, other=0.0)
            br_block_input_dot_grad = tl.sum(br_block_input*grad, axis=0)*br_block_mask

            dodx = dodx + 1 * br_block_input_dot_grad * (y - floor_y)
            dody = dody + 1 * br_block_input_dot_grad * (x - floor_x)
            dodw = dodw + br_block_input_dot_grad*br_weight

            dodbr = dods * br_weight
            tl.atomic_add(grad_input_ptr + br_block_offset, mask=br_block_mask  & block_mask, val=dodbr)
        dodx = dodx * weight
        dody = dody * weight
        tl.store(grad_weights_ptr + weights_offset + tl.arange(0, 1), dodw, mask=id_mask)
        tl.store(grad_deformable_ptr + deformable_offset + tl.arange(0, 1), dodx, mask=id_mask)
        tl.store(grad_deformable_ptr + deformable_offset + 1 + tl.arange(0, 1), dody, mask=id_mask)





