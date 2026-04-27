
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
