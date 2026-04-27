# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /root/.cache/inductor/ro/crofwq6okbite26skbfe6czxcuevp2xst3hx7b2pw3xdihhmtom6.py
# Topologically Sorted Source Nodes: [q, unsqueeze, mul, x2, neg, x1, cat, unsqueeze_1, mul_1, q_embed, q_embed_1, k, mul_2, x2_1, neg_1, x1_1, cat_1, mul_3, k_embed, k_embed_1], Original ATen: [aten._to_copy, aten.unsqueeze, aten.mul, aten.slice, aten.neg, aten.cat, aten.add]
# Source node to ATen node mapping:
#   cat => cat
#   cat_1 => cat_1
#   k => convert_element_type_1
#   k_embed => add_69
#   k_embed_1 => convert_element_type_3
#   mul => mul_12
#   mul_1 => mul_28
#   mul_2 => mul_35
#   mul_3 => mul_51
#   neg => neg
#   neg_1 => neg_1
#   q => convert_element_type
#   q_embed => add_40
#   q_embed_1 => convert_element_type_2
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
#   x1 => slice_1
#   x1_1 => slice_3
#   x2 => slice_2
#   x2_1 => slice_4
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s48, s32, s91][s32*s91, s91, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg5_1 : Tensor "f32[s48, s91][s91, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg7_1 : Tensor "f32[s48, s91][s91, 1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg4_1 : Tensor "bf16[s48, s32, s91][s32*s91, s91, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convert_element_type : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %unsqueeze : Tensor "f32[s48, 1, s91][s91, s91, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 1), kwargs = {})
#   %mul_12 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %unsqueeze), kwargs = {})
#   %slice_2 : Tensor "f32[s48, s32, s91 - ((s91//2))][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type, 2, %floordiv, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[s48, s32, s91 - ((s91//2))][s32*Max(1, s91 - ((s91//2))), Max(1, s91 - ((s91//2))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %slice_1 : Tensor "f32[s48, s32, (s91//2)][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type, 2, 0, %floordiv), kwargs = {})
#   %cat : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[s48, 1, s91][s91, s91, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, 1), kwargs = {})
#   %mul_28 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_1), kwargs = {})
#   %add_40 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_28), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.bfloat16), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg4_1, torch.float32), kwargs = {})
#   %mul_35 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %unsqueeze), kwargs = {})
#   %slice_4 : Tensor "f32[s48, s32, s91 - ((s91//2))][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 2, %floordiv, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "f32[s48, s32, s91 - ((s91//2))][s32*Max(1, s91 - ((s91//2))), Max(1, s91 - ((s91//2))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_4,), kwargs = {})
#   %slice_3 : Tensor "f32[s48, s32, (s91//2)][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_1, 2, 0, %floordiv), kwargs = {})
#   %cat_1 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_3], -1), kwargs = {})
#   %mul_51 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_1), kwargs = {})
#   %add_69 : Tensor "f32[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %mul_51), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s48, s32, s91][s32*s91, s91, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_69, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2,%convert_element_type_3
triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0 = async_compile.triton('triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '41663D836B68660E17620394D41E46BA52E5DD32AD3AC8DE56299BA5F621208E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % ks0)
    x2 = xindex // ks1
    x4 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + ks0*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (x0 + ks0*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = ks0 + (-1)*(ks0 // 2)
    tmp8 = tmp4 < tmp7
    tmp9 = tl.load(in_ptr0 + (ks0*x4 + (ks0 // 2) + (x0)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = -tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp7
    tmp15 = ks0
    tmp16 = tmp4 < tmp15
    tmp17 = tl.load(in_ptr0 + (ks0*x4 + (x0 + ((-1)*ks0) + (ks0 // 2))), tmp14 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tl.where(tmp8, tmp13, tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp3 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp2
    tmp29 = tl.load(in_ptr3 + (ks0*x4 + (ks0 // 2) + (x0)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = -tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp8, tmp31, tmp32)
    tmp34 = tl.load(in_ptr3 + (ks0*x4 + (x0 + ((-1)*ks0) + (ks0 // 2))), tmp14 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp14, tmp35, tmp36)
    tmp38 = tl.where(tmp8, tmp33, tmp37)
    tmp39 = tmp38 * tmp22
    tmp40 = tmp28 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tl.store(out_ptr1 + (x3), tmp41, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
        args.clear()
        s48 = arg0_1
        s32 = arg1_1
        s91 = arg2_1
        s38 = arg6_1
        assert_size_stride(arg3_1, (s48, s32, s91), (s32*s91, s91, 1))
        assert_size_stride(arg4_1, (s48, s32, s91), (s32*s91, s91, 1))
        assert_size_stride(arg5_1, (s48, s91), (s91, 1))
        assert_size_stride(arg7_1, (s48, s91), (s91, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            ps0 = s32*s91
            buf0 = empty_strided_cuda((s48, s32, s91), (s32*s91, s91, 1), torch.bfloat16)
            buf1 = empty_strided_cuda((s48, s32, s91), (s32*s91, s91, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [q, unsqueeze, mul, x2, neg, x1, cat, unsqueeze_1, mul_1, q_embed, q_embed_1, k, mul_2, x2_1, neg_1, x1_1, cat_1, mul_3, k_embed, k_embed_1], Original ATen: [aten._to_copy, aten.unsqueeze, aten.mul, aten.slice, aten.neg, aten.cat, aten.add]
            triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0_xnumel = s32*s48*s91
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0.run(arg3_1, arg5_1, arg7_1, arg4_1, buf0, buf1, s91, ps0, triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_0_xnumel, stream=stream0)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg7_1
        return (buf0, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 256
    arg1_1 = 8
    arg2_1 = 72
    arg3_1 = rand_strided((256, 8, 72), (576, 72, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((256, 8, 72), (576, 72, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((256, 72), (72, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = 1
    arg7_1 = rand_strided((256, 72), (72, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
