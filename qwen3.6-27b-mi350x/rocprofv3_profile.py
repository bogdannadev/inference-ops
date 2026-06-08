#!/usr/bin/env python3
"""
rocprofv3 Profiler for SGLang on AMD MI350X (gfx950)
Attaches to the running SGLang process and collects hardware PMCs.

Usage:
    # Quick overview (15 counters, minimal overhead)
    python3 rocprofv3_profile.py --pmcs quick

    # Focused profiles (run one at a time)
    python3 rocprofv3_profile.py --pmcs occupancy
    python3 rocprofv3_profile.py --pmcs cache
    python3 rocprofv3_profile.py --pmcs utilization
    python3 rocprofv3_profile.py --pmcs memory
    python3 rocprofv3_profile.py --pmcs stalls
    python3 rocprofv3_profile.py --pmcs mfma
    python3 rocprofv3_profile.py --pmcs kernel
    python3 rocprofv3_profile.py --pmcs sq

    # All profiles sequentially (useful for full analysis)
    python3 rocprofv3_profile.py --all-profiles --duration 15

    # Custom PMC set
    python3 rocprofv3_profile.py --custom-pmcs "GPU_UTIL,OccupancyPercent,MfmaFlopsBF16"

    # List all available PMCs by category
    python3 rocprofv3_profile.py --pmcs list
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROCPROFV3 = "/opt/rocm/bin/rocprofv3"

# ============================================================================
# PMC SETS BY CATEGORY
# ============================================================================

PMCS_QUICK = [
    "GPU_UTIL",
    "OccupancyPercent",
    "MeanOccupancyPerCU",
    "MeanOccupancyPerActiveCU",
    "MfmaFlops",
    "MfmaFlopsBF16",
    "MfmaUtil",
    "MemWrites32B",
    "MemUnitStalled",
    "GRBM_CPC_BUSY",
    "GRBM_CPF_BUSY",
    "GRBM_TC_BUSY",
    "CU_NUM",
    "SIMD_NUM",
    "SE_NUM",
]

PMCS_OCCUPANCY = [
    "GPU_UTIL",
    "OccupancyPercent",
    "MeanOccupancyPerCU",
    "MeanOccupancyPerActiveCU",
    "CU_NUM",
    "SIMD_NUM",
    "SE_NUM",
    "MAX_WAVE_SIZE",
    "SQ_WAVES",
    "SQ_WAVES_sum",
    "SQ_WAVES_EQ_64",
    "SQ_WAVES_LT_16",
    "SQ_WAVES_LT_32",
    "SQ_WAVES_LT_48",
    "SQ_WAVES_LT_64",
    "SQ_WAVES_RESTORED",
    "SQ_WAVES_SAVED",
    "SQ_LEVEL_WAVES",
    "SQ_BUSY_CYCLES",
    "SQ_BUSY_CU_CYCLES",
    "SQ_WAVE_CYCLES",
    "SQ_THREAD_CYCLES_VALU",
    "SQ_CYCLES",
    "SPI_CSN_WAVE",
    "SPI_CSN_NUM_THREADGROUPS",
    "SPI_CSN_BUSY",
    "SPI_CS0_WAVE",
    "SPI_CS1_WAVE",
    "SPI_CS2_WAVE",
    "SPI_CS3_WAVE",
    "SPI_CS0_NUM_THREADGROUPS",
    "SPI_CS1_NUM_THREADGROUPS",
    "SPI_CS2_NUM_THREADGROUPS",
    "SPI_CS3_NUM_THREADGROUPS",
    "SPI_CS0_BUSY",
    "SPI_CS1_BUSY",
    "SPI_CS2_BUSY",
    "SPI_CS3_BUSY",
    "SPI_CSQ_P0_OCCUPANCY",
    "SPI_CSQ_P1_OCCUPANCY",
    "SPI_CSQ_P2_OCCUPANCY",
    "SPI_CSQ_P3_OCCUPANCY",
    "SPI_CSQ_P0_Q0_OCCUPANCY",
    "SPI_CSQ_P0_Q1_OCCUPANCY",
    "SPI_CSQ_P0_Q2_OCCUPANCY",
    "SPI_CSQ_P0_Q3_OCCUPANCY",
    "SPI_CSQ_P1_Q0_OCCUPANCY",
    "SPI_CSQ_P1_Q1_OCCUPANCY",
    "SPI_CSQ_P1_Q2_OCCUPANCY",
    "SPI_CSQ_P1_Q3_OCCUPANCY",
]

PMCS_CACHE = [
    "LdsUtil",
    "LdsLatency",
    "LdsBankConflict",
    "LDSBankConflict",
    "SQ_LDS_BANK_CONFLICT",
    "SQ_LDS_ADDR_CONFLICT",
    "SQ_LDS_DATA_FIFO_FULL",
    "SQ_LDS_CMD_FIFO_FULL",
    "SQ_LDS_IDX_ACTIVE",
    "SQ_LDS_MEM_VIOLATIONS",
    "SQ_LDS_UNALIGNED_STALL",
    "SQ_LDS_ATOMIC_RETURN",
    "SQ_WAIT_INST_LDS",
    "SQ_INSTS_LDS",
    "SQ_INSTS_LDS_LOAD",
    "SQ_INSTS_LDS_STORE",
    "SQ_INSTS_LDS_ATOMIC",
    "SQ_INSTS_LDS_LOAD_BANDWIDTH",
    "SQ_INSTS_LDS_STORE_BANDWIDTH",
    "SQ_INSTS_LDS_ATOMIC_BANDWIDTH",
    "SQ_ACTIVE_INST_LDS",
    "SQ_INST_LEVEL_LDS",
    "SQC_ICACHE_HITS",
    "SQC_ICACHE_BUSY_CYCLES",
    "SQC_DCACHE_BUSY_CYCLES",
    "SQC_DCACHE_ATOMIC",
    "TCC_BUSY",
    "TCC_BUSY_avr",
    "TCC_BUSY_sum",
    "TCC_CC_REQ",
    "TCC_IB_REQ",
    "TCC_IB_STALL",
    "TCC_IB_STALL_sum",
    "TCC_TAG_STALL",
    "TCC_TAG_STALL_sum",
    "TCC_BUBBLE",
    "TCC_ALL_TC_OP_INV_EVICT",
    "TCC_ALL_TC_OP_INV_EVICT_sum",
    "TCC_ALL_TC_OP_WB_WRITEBACK",
    "TCC_ALL_TC_OP_WB_WRITEBACK_sum",
    "TCA_BUSY",
    "TCA_BUSY_sum",
    "TCP_UTCL1_TRANSLATION_HIT",
    "TCP_UTCL1_TRANSLATION_HIT_sum",
    "TCP_UTCL1_TRANSLATION_MISS",
    "TCP_UTCL1_TRANSLATION_MISS_sum",
    "TCP_CACHE_MISS",
    "SQC_TC_REQ",
    "SQC_TC_STALL",
    "SQC_TC_DATA_READ_REQ",
    "SQC_TC_DATA_WRITE_REQ",
    "SQC_TC_INST_REQ",
]

PMCS_UTILIZATION = [
    "GPU_UTIL",
    "SIMD_UTILIZATION",
    "VALUBusy",
    "VALUUtilization",
    "SALUBusy",
    "MfmaUtil",
    "OccupancyPercent",
    "MeanOccupancyPerCU",
    "MeanOccupancyPerActiveCU",
    "GRBM_COUNT",
    "GRBM_CPC_BUSY",
    "GRBM_CPF_BUSY",
    "GRBM_CP_BUSY",
    "GRBM_EA_BUSY",
    "GRBM_GUI_ACTIVE",
    "GRBM_SPI_BUSY",
    "GRBM_TA_BUSY",
    "GRBM_TC_BUSY",
    "GRBM_UTCL2_BUSY",
    "CPC_CPC_STAT_BUSY",
    "CPC_CPC_STAT_IDLE",
    "CPC_CPC_STAT_STALL",
    "CPF_CPF_STAT_BUSY",
    "CPF_CPF_STAT_IDLE",
    "CPF_CPF_STAT_STALL",
    "CPC_CPC_TCIU_BUSY",
    "CPC_CPC_TCIU_IDLE",
    "CPC_CPC_UTCL2IU_BUSY",
    "CPC_CPC_UTCL2IU_IDLE",
    "CPF_CPF_TCIU_BUSY",
    "CPF_CPF_TCIU_IDLE",
    "TA_TA_BUSY",
    "TA_TA_BUSY_sum",
    "TA_BUSY_avr",
    "TA_BUSY_max",
    "TA_BUSY_min",
    "TD_TD_BUSY",
    "TD_TD_BUSY_sum",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_VALU_MFMA_COEXEC_CYCLES",
    "SQ_INST_CYCLES_SALU",
    "SQ_INST_CYCLES_VMEM_WR",
    "SQ_ACTIVE_INST_FLAT",
    "SQ_ACTIVE_INST_VALU2",
]

PMCS_MEMORY = [
    "MemWrites32B",
    "MemUnitStalled",
    "BANDWIDTH_EA",
    "FETCH_SIZE",
    "FetchSize",
    "InstrFetchLatency",
    "VFetchInsts",
    "SQ_INSTS_FLAT_FLATSEG",
    "SQ_INSTS_FLAT_NO_LDS",
    "TA_BUFFER_READ_WAVEFRONTS",
    "TA_BUFFER_READ_WAVEFRONTS_sum",
    "TA_BUFFER_WRITE_WAVEFRONTS",
    "TA_BUFFER_WRITE_WAVEFRONTS_sum",
    "TA_BUFFER_TOTAL_CYCLES",
    "TA_BUFFER_TOTAL_CYCLES_sum",
    "TA_BUFFER_WAVEFRONTS",
    "TA_BUFFER_WAVEFRONTS_sum",
    "TA_BUFFER_COALESCED_READ_CYCLES",
    "TA_BUFFER_COALESCED_WRITE_CYCLES",
    "TA_BUFFER_COALESCED_WRITE_CYCLES_sum",
    "TA_BUFFER_COALESCEABLE_WAVEFRONTS",
    "TA_BUFFER_COALESCEABLE_WAVEFRONTS_sum",
    "TA_FLAT_READ_WAVEFRONTS",
    "TA_FLAT_READ_WAVEFRONTS_sum",
    "TA_FLAT_WRITE_WAVEFRONTS",
    "TA_FLAT_WRITE_WAVEFRONTS_sum",
    "TA_FLAT_WAVEFRONTS",
    "TA_FLAT_WAVEFRONTS_sum",
    "TA_FLAT_COALESCEABLE_WAVEFRONTS",
    "TA_FLAT_COALESCEABLE_WAVEFRONTS_sum",
    "TA_TOTAL_WAVEFRONTS",
    "TA_TOTAL_WAVEFRONTS_sum",
    "TA_ADDR_STALLED_BY_TC_CYCLES",
    "TA_ADDR_STALLED_BY_TC_CYCLES_sum",
    "TA_ADDR_STALLED_BY_TD_CYCLES",
    "TA_ADDR_STALLED_BY_TD_CYCLES_sum",
    "TA_DATA_STALLED_BY_TC_CYCLES",
    "TA_DATA_STALLED_BY_TC_CYCLES_sum",
    "TCC_EA0_RDREQ_DRAM",
    "TCC_EA0_RDREQ_DRAM_sum",
    "TCC_EA0_RDREQ_DRAM_32B_sum",
    "TCC_EA0_RDREQ_DRAM_CREDIT_STALL",
    "TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum",
    "TCC_EA0_RDREQ_GMI_CREDIT_STALL",
    "TCC_EA0_RDREQ_GMI_CREDIT_STALL_sum",
    "TCC_EA0_RDREQ_IO_CREDIT_STALL",
    "TCC_EA0_RDREQ_IO_CREDIT_STALL_sum",
    "TCC_EA0_WRREQ_DRAM",
    "TCC_EA0_WRREQ_DRAM_sum",
    "TCC_EA0_WRREQ_DRAM_CREDIT_STALL",
    "TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum",
    "TCC_EA0_WRREQ_GMI_CREDIT_STALL",
    "TCC_EA0_WRREQ_GMI_CREDIT_STALL_sum",
    "TCC_EA0_WRREQ_IO_CREDIT_STALL",
    "TCC_EA0_WRREQ_IO_CREDIT_STALL_sum",
    "TCC_EA0_WRREQ_STALL",
    "TCC_EA0_WRREQ_STALL_sum",
    "TCC_EA0_WRREQ_sum",
    "TCC_EA0_WRREQ_PROBE_COMMAND",
    "TCC_EA0_WRREQ_WRITE_DRAM_32B_sum",
    "TCC_TOO_MANY_EA_WRREQS_STALL",
    "TCC_TOO_MANY_EA_WRREQS_STALL_sum",
    "TCC_WRREQ_STALL_max",
    "TD_LOAD_WAVEFRONT",
    "TD_LOAD_WAVEFRONT_sum",
    "TD_STORE_WAVEFRONT",
    "TD_STORE_WAVEFRONT_sum",
    "TD_WRITE_ACKT_WAVEFRONT",
    "TD_WRITE_ACKT_WAVEFRONT_sum",
    "TD_COALESCABLE_WAVEFRONT",
    "TD_COALESCABLE_WAVEFRONT_sum",
    "SQ_VMEM_TA_ADDR_FIFO_FULL",
    "SQ_VMEM_TA_CMD_FIFO_FULL",
    "SQ_VMEM_WR_TA_DATA_FIFO_FULL",
    "RDC_OPS_32_PER_SIMDCYCLE",
    "RDC_OPS_64_PER_SIMDCYCLE",
    "RDC_OPS_16_PER_SIMDCYCLE",
]

PMCS_STALLS = [
    "MemUnitStalled",
    "SmemLatency",
    "CPC_CPC_STAT_STALL",
    "CPC_CPC_UTCL2IU_STALL",
    "CPC_UTCL1_STALL_ON_TRANSLATION",
    "CPC_STALLED_BY_SE0_SPI",
    "CPC_STALLED_BY_SE1_SPI",
    "CPC_STALLED_BY_SE2_SPI",
    "CPC_STALLED_BY_SE3_SPI",
    "CPC_CANE_STALL",
    "CPC_SYNC_FIFO_FULL",
    "CPC_SYNC_FIFO_FULL_LEVEL",
    "CPC_SYNC_WRREQ_FIFO_BUSY",
    "CPF_CPF_STAT_STALL",
    "CPF_CPF_TCIU_STALL",
    "CPF_CMP_UTCL1_STALL_ON_TRANSLATION",
    "SPI_RA_RES_STALL_CSN",
    "SPI_RA_TMP_STALL_CSN",
    "SPI_RA_WVLIM_STALL_CSN",
    "SPI_RA_TGLIM_CU_FULL_CSN",
    "SPI_RA_SGPR_SIMD_FULL_CSN",
    "SPI_RA_VGPR_SIMD_FULL_CSN",
    "SPI_RA_WAVE_SIMD_FULL_CSN",
    "SPI_RA_BAR_CU_FULL_CSN",
    "SPI_RA_BULKY_CU_FULL_CSN",
    "SPI_RA_LDS_CU_FULL_CSN",
    "SPI_RA_REQ_NO_ALLOC",
    "SPI_RA_REQ_NO_ALLOC_CSN",
    "SPI_CS0_CRAWLER_STALL",
    "SPI_CS1_CRAWLER_STALL",
    "SPI_CS2_CRAWLER_STALL",
    "SPI_CS3_CRAWLER_STALL",
    "SQ_LDS_UNALIGNED_STALL",
    "SQ_LDS_DATA_FIFO_FULL",
    "SQ_LDS_CMD_FIFO_FULL",
    "TD_SPI_STALL",
    "TD_SPI_STALL_sum",
    "TD_TC_STALL",
    "TD_TC_STALL_sum",
    "TCP_TCP_TA_ADDR_STALL_CYCLES",
    "TCP_TCP_TA_ADDR_STALL_CYCLES_sum",
    "TCP_TCP_TA_DATA_STALL_CYCLES",
    "TCP_TCP_TA_DATA_STALL_CYCLES_sum",
    "TCP_TCP_TA_DATA_STALL_CYCLES_max",
    "TCP_PENDING_STALL_CYCLES",
    "TCP_PENDING_STALL_CYCLES_sum",
    "TCP_LFIFO_STALL_CYCLES",
    "TCP_LFIFO_STALL_CYCLES_sum",
    "TCP_RFIFO_STALL_CYCLES",
    "TCP_RFIFO_STALL_CYCLES_sum",
    "TCP_TCR_RDRET_STALL",
    "TCP_TCR_RDRET_STALL_sum",
    "TCP_TCR_TCP_STALL_CYCLES",
    "TCP_TCR_TCP_STALL_CYCLES_sum",
    "TCP_TD_TCP_STALL_CYCLES",
    "TCP_TD_TCP_STALL_CYCLES_sum",
    "TCP_UTCL1_STALL_LFIFO_NO_RES",
    "TCP_UTCL1_STALL_LFIFO_NO_RES_sum",
    "TCP_UTCL1_STALL_INFLIGHT_MAX",
    "TCP_UTCL1_STALL_INFLIGHT_MAX_sum",
    "TCP_UTCL1_STALL_LRU_INFLIGHT",
    "TCP_UTCL1_STALL_MULTI_MISS",
    "TCP_UTCL1_STALL_MULTI_MISS_sum",
    "TCP_UTCL1_STALL_LFIFO_NOT_RES",
    "TCP_UTCL1_STALL_UTCL2_REQ_OUT_OF_CREDITS",
    "TCP_UTCL1_STALL_UTCL2_REQ_OUT_OF_CREDITS_sum",
    "TCP_UTCL1_LFIFO_FULL",
    "TCP_UTCL1_LFIFO_FULL_sum",
    "TCP_UTCL1_THRASHING_STALL",
    "TCP_UTCL1_THRASHING_STALL_sum",
    "TCP_UTCL1_SERIALIZATION_STALL",
    "TCP_UTCL1_TRANSLATION_MISS_UNDER_MISS",
    "TCP_UTCL1_TRANSLATION_MISS_UNDER_MISS_sum",
    "TCP_READ_TAGCONFLICT_STALL_CYCLES",
    "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum",
    "TCP_WRITE_TAGCONFLICT_STALL_CYCLES",
    "TCP_WRITE_TAGCONFLICT_STALL_CYCLES_sum",
    "TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES",
    "TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES_sum",
    "TCP_TAGRAM0_REQ",
    "TCP_CLIENT_UTCL1_INFLIGHT",
    "TCP_CLIENT_UTCL1_INFLIGHT_sum",
    "TCP_UTCL1_REQUEST",
    "TCP_UTCL1_REQUEST_sum",
    "TCP_UTCL1_PERMISSION_MISS",
    "TCP_UTCL1_PERMISSION_MISS_sum",
    "TCP_TCP_LATENCY_sum",
    "TCP_TA_TCP_STATE_READ",
    "TCP_TA_TCP_STATE_READ_sum",
    "TCP_GATE_EN1",
    "TCP_GATE_EN2_sum",
    "TCP_CP_TCP_INVALIDATE_VOL",
]

PMCS_MFMA = [
    "MfmaFlops",
    "MfmaFlopsBF16",
    "MfmaFlopsF16",
    "MfmaFlopsF32",
    "MfmaFlopsF64",
    "MfmaUtil",
    "SALUBusy",
    "VALUBusy",
    "VALUUtilization",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU_MFMA_BF16",
    "SQ_INSTS_VALU_MFMA_F16",
    "SQ_INSTS_VALU_MFMA_F32",
    "SQ_INSTS_VALU_MFMA_F64",
    "SQ_INSTS_VALU_MFMA_F6F4",
    "SQ_INSTS_VALU_MFMA_I8",
    "SQ_INSTS_VALU_MFMA_F8",
    "SQ_INSTS_VALU_MFMA_XF32",
    "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F64",
    "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
    "SQ_INSTS_VALU_MFMA_MOPS_I8",
    "SQ_INSTS_VALU_MFMA_MOPS_F8",
    "SQ_INSTS_VALU_MFMA_MOPS_XF32",
    "SQ_INSTS_VALU_MUL_F16",
    "SQ_INSTS_VALU_FLOPS_FP16",
    "SQ_INSTS_VALU_FLOPS_FP16_TRANS",
    "SQ_INSTS_VALU_FLOPS_FP32",
    "SQ_INSTS_VALU_FLOPS_FP32_TRANS",
    "SQ_INSTS_VALU_FLOPS_FP64",
    "SQ_INSTS_VALU_FLOPS_FP64_TRANS",
    "SQ_INSTS_VALU_IOPS",
    "SQ_INSTS_SALU",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_VALU_MFMA_COEXEC_CYCLES",
    "SQ_INST_CYCLES_SALU",
    "SQ_THREAD_CYCLES_VALU",
    "TOTAL_64_OPS",
]

PMCS_KERNEL = [
    "GPU_UTIL",
    "SIMD_UTILIZATION",
    "CU_NUM",
    "SIMD_NUM",
    "SE_NUM",
    "SQ_INSTS",
    "SQ_ITEMS",
    "SQ_IFETCH",
    "SQ_IFETCH_LEVEL",
    "SQ_VSKIPPED",
    "SQ_INSTS_FLAT_FLATSEG",
    "SQ_INSTS_FLAT_NO_LDS",
    "SQ_INSTS_LDS",
    "SQ_INSTS_LDS_LOAD",
    "SQ_INSTS_LDS_STORE",
    "SQ_INSTS_LDS_ATOMIC",
    "SQ_ACTIVE_INST_FLAT",
    "SQ_ACTIVE_INST_LDS",
    "SQ_ACTIVE_INST_VALU2",
    "SQ_BUSY_CYCLES",
    "SQ_BUSY_CU_CYCLES",
    "SQ_CYCLES",
    "SQ_WAVES",
    "SQ_WAVES_sum",
    "SQ_WAVE_CYCLES",
    "SQ_LEVEL_WAVES",
    "SQ_WAVES_EQ_64",
    "SQ_WAVES_LT_16",
    "SQ_WAVES_LT_32",
    "SQ_WAVES_LT_48",
    "SQ_WAVES_LT_64",
    "SQ_WAVES_RESTORED",
    "SQ_WAVES_SAVED",
    "OccupancyPercent",
    "MeanOccupancyPerCU",
    "MeanOccupancyPerActiveCU",
    "MAX_WAVE_SIZE",
    "FETCH_SIZE",
    "FetchSize",
    "InstrFetchLatency",
    "VFetchInsts",
    "CPC_ALWAYS_COUNT",
    "CPC_CPC_STAT_BUSY",
    "CPC_CPC_STAT_IDLE",
    "CPC_CPC_STAT_STALL",
    "CPF_CPF_STAT_BUSY",
    "CPF_CPF_STAT_IDLE",
    "CPF_CPF_STAT_STALL",
    "GRBM_COUNT",
    "GRBM_CPC_BUSY",
    "GRBM_CPF_BUSY",
    "GRBM_SPI_BUSY",
    "SPI_CSN_BUSY",
    "SPI_CSN_WAVE",
    "SPI_CSN_NUM_THREADGROUPS",
    "SPI_CSN_WINDOW_VALID",
    "SPI_CS0_BUSY",
    "SPI_CS0_WAVE",
    "SPI_CS0_NUM_THREADGROUPS",
    "SPI_CS0_WINDOW_VALID",
    "SPI_CS1_BUSY",
    "SPI_CS1_WAVE",
    "SPI_CS1_NUM_THREADGROUPS",
    "SPI_CS1_WINDOW_VALID",
    "SPI_CS2_BUSY",
    "SPI_CS2_WAVE",
    "SPI_CS2_NUM_THREADGROUPS",
    "SPI_CS2_WINDOW_VALID",
    "SPI_CS3_BUSY",
    "SPI_CS3_WAVE",
    "SPI_CS3_NUM_THREADGROUPS",
    "SPI_CS3_WINDOW_VALID",
    "SPI_SWC_CSC_WR",
    "SPI_VWC_CSC_WR",
    "SPI_VWC0_VDATA_VALID_WR",
    "SPI_VWC1_VDATA_VALID_WR",
    "SPI_CSC_WAVE_CNT_BUSY",
    "SPI_CS0_EVENT_WAVE",
    "SPI_CS1_EVENT_WAVE",
    "SPI_CS2_EVENT_WAVE",
    "SPI_CS3_EVENT_WAVE",
]

PMCS_SQ = [
    "SQ_INSTS",
    "SQ_ITEMS",
    "SQ_IFETCH",
    "SQ_IFETCH_LEVEL",
    "SQ_VSKIPPED",
    "SQ_BUSY_CYCLES",
    "SQ_BUSY_CU_CYCLES",
    "SQ_CYCLES",
    "SQ_WAVES",
    "SQ_WAVES_sum",
    "SQ_WAVE_CYCLES",
    "SQ_LEVEL_WAVES",
    "SQ_WAVES_EQ_64",
    "SQ_WAVES_LT_16",
    "SQ_WAVES_LT_32",
    "SQ_WAVES_LT_48",
    "SQ_WAVES_LT_64",
    "SQ_WAVES_RESTORED",
    "SQ_WAVES_SAVED",
    "SQ_INSTS_FLAT_FLATSEG",
    "SQ_INSTS_FLAT_NO_LDS",
    "SQ_INSTS_LDS",
    "SQ_INSTS_LDS_LOAD",
    "SQ_INSTS_LDS_STORE",
    "SQ_INSTS_LDS_ATOMIC",
    "SQ_INSTS_LDS_LOAD_BANDWIDTH",
    "SQ_INSTS_LDS_STORE_BANDWIDTH",
    "SQ_INSTS_LDS_ATOMIC_BANDWIDTH",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SENDMSG",
    "SQ_INSTS_VALU_IOPS",
    "SQ_INSTS_VALU_FLOPS_FP16",
    "SQ_INSTS_VALU_FLOPS_FP32",
    "SQ_INSTS_VALU_FLOPS_FP64",
    "SQ_INSTS_VALU_MFMA_BF16",
    "SQ_INSTS_VALU_MFMA_F32",
    "SQ_INSTS_VALU_MFMA_F64",
    "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "SQ_INSTS_VALU_MFMA_MOPS_F64",
    "SQ_ACTIVE_INST_FLAT",
    "SQ_ACTIVE_INST_LDS",
    "SQ_ACTIVE_INST_VALU2",
    "SQ_INST_LEVEL_LDS",
    "SQ_INST_CYCLES_SALU",
    "SQ_INST_CYCLES_VMEM_WR",
    "SQ_THREAD_CYCLES_VALU",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_VALU_MFMA_COEXEC_CYCLES",
    "SQ_LDS_BANK_CONFLICT",
    "SQ_LDS_ADDR_CONFLICT",
    "SQ_LDS_DATA_FIFO_FULL",
    "SQ_LDS_CMD_FIFO_FULL",
    "SQ_LDS_IDX_ACTIVE",
    "SQ_LDS_MEM_VIOLATIONS",
    "SQ_LDS_UNALIGNED_STALL",
    "SQ_LDS_ATOMIC_RETURN",
    "SQ_WAIT_INST_LDS",
    "SQ_VMEM_TA_ADDR_FIFO_FULL",
    "SQ_VMEM_TA_CMD_FIFO_FULL",
    "SQ_VMEM_WR_TA_DATA_FIFO_FULL",
]

PMC_SETS = {
    "quick":     ("Quick Overview (15 counters, minimal overhead)", PMCS_QUICK),
    "occupancy": ("Wave Occupancy & CU Utilization", PMCS_OCCUPANCY),
    "cache":     ("L1/L2/L3 Cache, LDS, TCC, TCP", PMCS_CACHE),
    "utilization": ("GPU/SIMD/VALU/SALU/MFMA Utilization", PMCS_UTILIZATION),
    "memory":    ("Memory Bandwidth, TA/TCC/TCP, Coalescing", PMCS_MEMORY),
    "stalls":    ("All Stall Counters (SPI, CPC, CPF, TCC, TCP, TD)", PMCS_STALLS),
    "mfma":      ("MFMA FLOPS by Precision, VALU/SALU", PMCS_MFMA),
    "kernel":    ("Kernel Dispatch, Instructions, Waves, SPI", PMCS_KERNEL),
    "sq":        ("SQ-Level: Instructions, LDS, MFMA, Waves", PMCS_SQ),
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def find_sglang_pid():
    result = subprocess.run(
        ["amd-smi", "process", "--json"],
        capture_output=True, text=True, timeout=10
    )
    try:
        data = json.loads(result.stdout)
        best_pid = None
        best_name = None
        best_vram = 0
        for gpu in data:
            for p in gpu.get("process_list", []):
                info = p.get("process_info", {})
                name = info.get("name", "")
                pid = info.get("pid", 0)
                mem = info.get("mem_usage", {})
                vram = mem.get("value", 0) if isinstance(mem, dict) else 0
                if vram > best_vram:
                    best_vram = vram
                    best_pid = pid
                    best_name = name
        if best_pid and best_vram > 1e9:
            return best_pid, best_name, best_vram
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None, None, 0

def run_profile(pid, duration, pmcs, output_dir, label=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    output_path = os.path.join(output_dir, f"rocprof_{ts}{suffix}")

    cmd = [
        ROCPROFV3,
        "--attach", str(pid),
        "--attach-duration-msec", str(duration * 1000),
        "--pmc"] + list(pmcs) + [
        "--output-directory", output_dir,
        "--output-file", output_path,
        "--output-format", "json",
        "--summary",
        "--summary-units", "msec",
    ]

    print(f"\n{'='*60}")
    print(f"  Profile: {label or 'custom'}")
    print(f"  Attaching to PID {pid} for {duration}s")
    print(f"  PMCs: {len(pmcs)}")
    print(f"  Output: {output_path}.json")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 120)
    elapsed = time.time() - start

    print(f"\n  Completed in {elapsed:.1f}s (rc={result.returncode})")

    if result.stderr and result.returncode != 0:
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-5:]:
            print(f"  ERR: {line}")

    output_file = f"{output_path}.json"
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"  File: {file_size / 1024 / 1024:.1f} MB")
        try:
            with open(output_file) as f:
                data = json.load(f)
            print_analysis(data, label)
        except json.JSONDecodeError:
            print("  Could not parse output JSON")
    else:
        print(f"  Output file not found")
        for f in sorted(os.listdir(output_dir)):
            if ts in f:
                sz = os.path.getsize(os.path.join(output_dir, f))
                print(f"  Found: {f} ({sz/1024:.0f} KB)")

    return output_file

def print_analysis(data, label):
    print(f"\n  --- {label} Analysis ---")

    if not isinstance(data, dict):
        return

    for section, content in data.items():
        if not isinstance(content, dict):
            continue

        key_metrics = {}
        for k, v in content.items():
            if isinstance(v, (int, float)):
                key_metrics[k] = v

        if not key_metrics:
            continue

        print(f"\n  [{section}]")
        for k, v in sorted(key_metrics.items(), key=lambda x: x[0]):
            if isinstance(v, float):
                if abs(v) > 1e9:
                    print(f"    {k}: {v/1e9:.2f}B")
                elif abs(v) > 1e6:
                    print(f"    {k}: {v/1e6:.2f}M")
                elif abs(v) > 1e3:
                    print(f"    {k}: {v/1e3:.2f}K")
                else:
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

def list_available_pmcs():
    result = subprocess.run(
        ["/opt/rocm/bin/rocprofv3-avail", "list"],
        capture_output=True, text=True, timeout=10
    )
    lines = result.stdout.strip().split("\n")

    categories = {
        "Occupancy & Waves": [],
        "Cache (LDS/TCC/TCP)": [],
        "Utilization": [],
        "Memory & Bandwidth": [],
        "Stalls": [],
        "MFMA & VALU": [],
        "Kernel & SQ": [],
        "SPI & CSQ": [],
        "TA & TD": [],
        "CPC & CPF": [],
        "GRBM": [],
        "Other": [],
    }

    in_pmc = False
    for line in lines:
        if "PMC" in line and ":" in line:
            in_pmc = True
            continue
        if not in_pmc or not line.strip():
            continue
        for token in line.split():
            token = token.strip()
            if not token:
                continue
            if "Occupancy" in token or "WAVE" in token or "SQ_WAVES" in token or "MeanOccupancy" in token or "MAX_WAVE" in token:
                categories["Occupancy & Waves"].append(token)
            elif "Lds" in token or "LDS" in token or "TCC" in token or "TCP" in token or "CACHE" in token or "ICACHE" in token or "DCACHE" in token:
                categories["Cache (LDS/TCC/TCP)"].append(token)
            elif "UTIL" in token or "Busy" in token or "VALU" in token or "SALU" in token or "MfmaUtil" in token or "GPU_UTIL" in token or "SIMD_UTIL" in token:
                categories["Utilization"].append(token)
            elif "Mem" in token or "BANDWIDTH" in token or "FETCH" in token or "Fetch" in token or "TA_BUFFER" in token or "TA_FLAT" in token or "TD_" in token or "RDC_" in token or "TCC_EA" in token:
                categories["Memory & Bandwidth"].append(token)
            elif "STALL" in token or "STALL" in token or "STALL" in token or "CRAWLER" in token or "TGLIM" in token or "WVLIM" in token or "FULL" in token or "CONFLICT" in token:
                categories["Stalls"].append(token)
            elif "Mfma" in token or "MFMA" in token or "FLOP" in token or "MOPS" in token or "SALU" in token:
                categories["MFMA & VALU"].append(token)
            elif "SQ_" in token or "IFETCH" in token or "INST" in token or "ITEMS" in token or "CYCLES" in token:
                categories["Kernel & SQ"].append(token)
            elif "SPI_" in token or "CSQ_" in token or "CSC_" in token or "VWC_" in token:
                categories["SPI & CSQ"].append(token)
            elif "TA_" in token or "TD_" in token:
                categories["TA & TD"].append(token)
            elif "CPC_" in token or "CPF_" in token:
                categories["CPC & CPF"].append(token)
            elif "GRBM_" in token:
                categories["GRBM"].append(token)
            else:
                categories["Other"].append(token)

    for cat, pmcs in categories.items():
        unique = sorted(set(pmcs))
        if unique:
            print(f"\n{cat} ({len(unique)}):")
            for p in unique:
                print(f"  {p}")

def main():
    parser = argparse.ArgumentParser(
        description="rocprofv3 Profiler for SGLang on AMD MI350X (gfx950)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PMC Sets:
  quick        15 counters - GPU_UTIL, Occupancy, MFMA, Memory, GRBM
  occupancy    50 counters - Wave counts, CU occupancy, SPI queue occupancy
  cache        50 counters - LDS, TCC, TCP, cache hits/misses, bank conflicts
  utilization  45 counters - GPU/SIMD/VALU/SALU/MFMA busy/idle/stall
  memory       60 counters - Bandwidth, TA/TCC/TCP, coalescing, DRAM credits
  stalls       70 counters - All stall sources: SPI_RA, CPC, CPF, TCC, TCP, TD
  mfma         35 counters - MFMA FLOPS by precision, VALU/SALU cycles
  kernel       60 counters - Instructions, waves, dispatch, SPI activity
  sq           55 counters - SQ-level: instructions, LDS, MFMA, waves, FIFOs

Examples:
  python3 rocprofv3_profile.py --pmcs quick --duration 15
  python3 rocprofv3_profile.py --pmcs occupancy --duration 30
  python3 rocprofv3_profile.py --all-profiles --duration 15
  python3 rocprofv3_profile.py --custom-pmcs "GPU_UTIL,OccupancyPercent,MfmaFlopsBF16"
  python3 rocprofv3_profile.py --pmcs list
        """
    )
    parser.add_argument("--duration", type=int, default=30,
                        help="Profiling duration in seconds per profile (default: 30)")
    parser.add_argument("--output", type=str, default="/tmp/rocprof_output",
                        help="Output directory (default: /tmp/rocprof_output)")
    parser.add_argument("--pmcs", type=str,
                        choices=list(PMC_SETS.keys()) + ["list"],
                        default="quick",
                        help="PMC set to collect (default: quick)")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Run all profiles sequentially")
    parser.add_argument("--pid", type=int, default=0,
                        help="Target PID (auto-detect SGLang if 0)")
    parser.add_argument("--custom-pmcs", type=str, default="",
                        help="Comma-separated custom PMC list")
    args = parser.parse_args()

    if args.pmcs == "list":
        list_available_pmcs()
        return

    pid = args.pid
    if not pid:
        pid, name, vram = find_sglang_pid()
        if not pid:
            print("ERROR: Could not find SGLang process. Use --pid to specify.")
            sys.exit(1)
        print(f"Found SGLang: {name} (PID {pid}, VRAM: {vram/1e9:.1f} GB)")

    os.makedirs(args.output, exist_ok=True)

    if args.custom_pmcs:
        pmcs = [p.strip() for p in args.custom_pmcs.split(",")]
        run_profile(pid, args.duration, pmcs, args.output, "custom")
    elif args.all_profiles:
        print(f"\nRunning all {len(PMC_SETS)} profiles ({args.duration}s each)...")
        for name, (desc, pmcs) in PMC_SETS.items():
            print(f"\n{'#'*60}")
            print(f"# {name.upper()}: {desc} ({len(pmcs)} PMCs)")
            print(f"{'#'*60}")
            run_profile(pid, args.duration, pmcs, args.output, name)
            time.sleep(2)
        print(f"\n\nAll profiles complete. Output: {args.output}/")
        files = sorted([f for f in os.listdir(args.output) if f.startswith("rocprof_")])
        print(f"Files: {len(files)}")
        for f in files:
            sz = os.path.getsize(os.path.join(args.output, f))
            print(f"  {f} ({sz/1024:.0f} KB)")
    else:
        desc, pmcs = PMC_SETS[args.pmcs]
        print(f"\nProfile: {args.pmcs} - {desc} ({len(pmcs)} PMCs)")
        run_profile(pid, args.duration, pmcs, args.output, args.pmcs)

if __name__ == "__main__":
    main()
