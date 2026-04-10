// ============================================================================
// GPU Sort Pipeline Profiler — Per-kernel timing & bandwidth analysis
//
// Build:
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_80 -lineinfo \
//        -Iinclude experiments/profile_sort.cu src/run_generation.cu \
//        src/merge.cu -o profile_sort
//
// Run:
//   ./profile_sort [NUM_RECORDS]           (default 10M, K-way)
//   ./profile_sort 10000000 --2way         (force 2-way merge path)
// ============================================================================
#include "record.cuh"
#include "ovc.cuh"
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdlib>

// ── Merge descriptors (match merge.cu) ───────────────────────────────
struct PairDesc2Way {
    uint64_t a_byte_offset; int a_count;
    uint64_t b_byte_offset; int b_count;
    uint64_t out_byte_offset; int first_block;
};
static constexpr int KWAY_K = 8;
struct KWayPartition {
    int src_rec_start[KWAY_K], src_rec_count[KWAY_K];
    uint64_t src_byte_off[KWAY_K], out_byte_offset;
    int total_records;
};

// ── External kernel launchers ────────────────────────────────────────
extern "C" void launch_run_generation(const uint8_t*, uint8_t*, uint32_t*,
    uint64_t, SparseEntry*, int*, int, cudaStream_t);
extern "C" void launch_merge_2way(const uint8_t*, uint8_t*,
    const PairDesc2Way*, int, int, cudaStream_t);
extern "C" void launch_merge_kway(const uint8_t*, uint8_t*,
    const KWayPartition*, int, int, cudaStream_t);

struct Run { uint64_t byte_offset, num_records; };
struct PassStat {
    int pass, in_runs, out_runs, blocks;
    float ms;
    double bw_gbs, bw_pct, copy_pct, mrec_per_sec;
};

// ── Copy bandwidth ceiling ───────────────────────────────────────────
__global__ void copy_kern(const uint4* __restrict__ s, uint4* __restrict__ d, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}
static double measure_copy_bw(uint64_t bytes) {
    uint8_t *a, *b;
    CUDA_CHECK(cudaMalloc(&a, bytes)); CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 0xAB, bytes));
    uint64_t n4 = bytes / sizeof(uint4);
    int thr = 256, blk = (int)((n4 + thr - 1) / thr);
    copy_kern<<<blk,thr>>>((uint4*)a,(uint4*)b,n4);
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t; t.begin();
    copy_kern<<<blk,thr>>>((uint4*)a,(uint4*)b,n4);
    float ms = t.end();
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    return (2.0 * bytes) / (ms * 1e6);
}

// ── K-way helpers ────────────────────────────────────────────────────
static int max_rpp() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    return std::min((int)p.sharedMemPerMultiprocessor, 99*1024) / (2*RECORD_SIZE);
}
static void build_kway_parts(const std::vector<Run>& grp, int K, int mrpp,
                              std::vector<KWayPartition>& out, uint64_t obase) {
    uint64_t tot = 0;
    for (auto& r : grp) tot += r.num_records;
    int np = std::max((int)((tot + mrpp - 1) / mrpp), 64);
    int rpp = (int)((tot + np - 1) / np);
    out.resize(np);
    for (int p = 0; p < np; p++) {
        auto& kp = out[p];
        uint64_t ps = (uint64_t)p*rpp, pe = std::min(ps+(uint64_t)rpp, tot);
        kp.total_records = (int)(pe-ps);
        kp.out_byte_offset = obase + ps*RECORD_SIZE;
        uint64_t rem = kp.total_records;
        for (int k = 0; k < K; k++) {
            uint64_t rn = grp[k].num_records;
            int cnt = (k==K-1) ? (int)rem : std::min((int)rem,
                      (int)((rn*kp.total_records+tot-1)/tot));
            cnt = std::min(cnt,(int)rn);
            uint64_t ss = (rn*p)/np;
            int sc = std::min((uint64_t)cnt, rn-ss);
            kp.src_rec_start[k]=(int)ss; kp.src_rec_count[k]=sc;
            kp.src_byte_off[k]=grp[k].byte_offset; rem-=sc;
        }
        for (int k = K; k < KWAY_K; k++)
            kp.src_rec_start[k]=kp.src_rec_count[k]=0, kp.src_byte_off[k]=0;
    }
}

// ════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    uint64_t num_records = 10000000;
    bool use_2way = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--2way")) use_2way = true;
        else num_records = strtoull(argv[i], nullptr, 10);
    }
    uint64_t total_bytes = num_records * RECORD_SIZE;
    int num_runs = (int)((num_records + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK);
    int mrpp = max_rpp();

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp props; cudaGetDeviceProperties(&props, dev);
    double peak_bw = 2.0 * props.memoryClockRate * (props.memoryBusWidth/8) / 1e6;
    int fanin = use_2way ? 2 : KWAY_K;

    printf("\n%s\n  GPU Sort Profile\n%s\n",
           "═══════════════════════════════════════════════════",
           "═══════════════════════════════════════════════════");
    printf("GPU: %s\nPeak HBM BW: %.0f GB/s\n", props.name, peak_bw);
    printf("Data: %lu records x %d B = %.2f GB\n", num_records, RECORD_SIZE, total_bytes/1e9);
    printf("Strategy: %s\n\n", use_2way ? "2-way merge path" : "8-way smem merge tree");

    printf("Measuring copy bandwidth ceiling...\n");
    double copy_bw = measure_copy_bw(total_bytes);
    printf("  Copy ceiling: %.1f GB/s (read+write)\n\n", copy_bw);

    // ── Generate random data ──
    uint8_t* h = (uint8_t*)malloc(total_bytes);
    srand(42);
    for (uint64_t i = 0; i < num_records; i++) {
        uint8_t* r = h + i*RECORD_SIZE;
        for (int b = 0; b < KEY_SIZE; b++) r[b] = (uint8_t)(rand()&0xFF);
        memset(r+KEY_SIZE, 0, VALUE_SIZE);
    }
    uint8_t* d_data; CUDA_CHECK(cudaMalloc(&d_data, total_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h, total_bytes, cudaMemcpyHostToDevice));
    free(h);

    // ── Phase 1: Run Generation ──
    uint8_t* d_sorted; uint32_t* d_ovc; SparseEntry* d_sp; int* d_spc;
    CUDA_CHECK(cudaMalloc(&d_sorted, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_ovc, num_records*sizeof(uint32_t)));
    int msp = num_runs*((RECORDS_PER_BLOCK+SPARSE_INDEX_STRIDE-1)/SPARSE_INDEX_STRIDE);
    CUDA_CHECK(cudaMalloc(&d_sp, std::max(1,msp)*(int)sizeof(SparseEntry)));
    CUDA_CHECK(cudaMalloc(&d_spc, std::max(1,num_runs)*(int)sizeof(int)));

    launch_run_generation(d_data,d_sorted,d_ovc,num_records,d_sp,d_spc,num_runs,0);
    CUDA_CHECK(cudaDeviceSynchronize()); // warmup

    GpuTimer timer; timer.begin();
    launch_run_generation(d_data,d_sorted,d_ovc,num_records,d_sp,d_spc,num_runs,0);
    float rg_ms = timer.end();
    double rg_bw = (2.0*total_bytes)/(rg_ms*1e6);

    printf("Phase 1: Run Generation\n");
    printf("  Time:        %.2f ms\n", rg_ms);
    printf("  Throughput:  %.1f GB/s (%.2f%% of peak)\n", rg_bw, 100.0*rg_bw/peak_bw);
    printf("  Runs:        %d (each %d records)\n", num_runs, RECORDS_PER_BLOCK);
    printf("  Records/sec: %.0fM\n", num_records/(rg_ms*1e3));
    printf("  Blocks: %d, Threads/block: %d\n", num_runs, BLOCK_THREADS);
    printf("  SMEM/block:  %.1f KB\n\n",
           RECORDS_PER_BLOCK*(sizeof(SortKey)+sizeof(int))/1024.0);

    CUDA_CHECK(cudaFree(d_ovc)); CUDA_CHECK(cudaFree(d_sp)); CUDA_CHECK(cudaFree(d_spc));

    std::vector<Run> runs(num_runs);
    for (int i = 0; i < num_runs; i++) {
        runs[i] = {(uint64_t)i*RECORDS_PER_BLOCK*RECORD_SIZE,
                   std::min((uint64_t)RECORDS_PER_BLOCK,
                            num_records-(uint64_t)i*RECORDS_PER_BLOCK)};
    }

    // ── Phase 2: Merge ──
    uint8_t* d_mbuf; CUDA_CHECK(cudaMalloc(&d_mbuf, total_bytes));
    uint8_t* d_src = d_sorted, *d_dst = d_mbuf;
    int exp_passes = (int)ceil(log((double)num_runs)/log((double)fanin));
    printf("Phase 2: Merge (%d-way, %d passes expected)\n", fanin, exp_passes);

    std::vector<PassStat> stats;
    double cum_traffic = 2.0*total_bytes;
    int pass = 0;

    while (runs.size() > 1) {
        pass++;
        int cur = (int)runs.size();
        float pass_ms = 0; int pass_blk = 0;
        std::vector<Run> nruns; uint64_t ooff = 0;

        if (use_2way) {
            int np = cur/2; bool left = (cur%2==1);
            std::vector<PairDesc2Way> pairs(np);
            int tblk = 0;
            for (int p = 0; p < np; p++) {
                int pt=(int)(runs[2*p].num_records+runs[2*p+1].num_records);
                int pb=(pt+8*256-1)/(8*256);
                pairs[p]={runs[2*p].byte_offset,(int)runs[2*p].num_records,
                          runs[2*p+1].byte_offset,(int)runs[2*p+1].num_records,ooff,tblk};
                tblk+=pb; ooff+=(uint64_t)pt*RECORD_SIZE;
            }
            PairDesc2Way* dp; CUDA_CHECK(cudaMalloc(&dp,np*sizeof(PairDesc2Way)));
            CUDA_CHECK(cudaMemcpy(dp,pairs.data(),np*sizeof(PairDesc2Way),cudaMemcpyHostToDevice));
            timer.begin();
            launch_merge_2way(d_src,d_dst,dp,np,tblk,0);
            if (left) { auto& rl=runs[cur-1];
                CUDA_CHECK(cudaMemcpyAsync(d_dst+ooff,d_src+rl.byte_offset,
                           rl.num_records*RECORD_SIZE,cudaMemcpyDeviceToDevice,0)); }
            pass_ms=timer.end(); pass_blk=tblk;
            CUDA_CHECK(cudaFree(dp));
            uint64_t noff=0;
            for (int p=0;p<np;p++){uint64_t c=runs[2*p].num_records+runs[2*p+1].num_records;
                nruns.push_back({noff,c}); noff+=c*RECORD_SIZE;}
            if (left) nruns.push_back({ooff,runs[cur-1].num_records});
        } else {
            int gs=std::min(KWAY_K,cur), ng=(cur+gs-1)/gs;
            for (int g=0;g<ng;g++){
                int gst=g*gs, ge=std::min(gst+gs,cur), gsz=ge-gst;
                if (gsz==1){ auto& r=runs[gst];
                    CUDA_CHECK(cudaMemcpy(d_dst+ooff,d_src+r.byte_offset,
                               r.num_records*RECORD_SIZE,cudaMemcpyDeviceToDevice));
                    nruns.push_back({ooff,r.num_records}); ooff+=r.num_records*RECORD_SIZE;
                    continue; }
                std::vector<Run> grp(runs.begin()+gst, runs.begin()+ge);
                std::vector<KWayPartition> parts;
                build_kway_parts(grp,gsz,mrpp,parts,ooff);
                int mr=0; for (auto& p:parts) mr=std::max(mr,p.total_records);
                KWayPartition* dp;
                CUDA_CHECK(cudaMalloc(&dp,parts.size()*sizeof(KWayPartition)));
                CUDA_CHECK(cudaMemcpy(dp,parts.data(),parts.size()*sizeof(KWayPartition),
                           cudaMemcpyHostToDevice));
                timer.begin();
                launch_merge_kway(d_src,d_dst,dp,(int)parts.size(),mr,0);
                pass_ms+=timer.end(); pass_blk+=(int)parts.size();
                CUDA_CHECK(cudaFree(dp));
                uint64_t merged=0; for(auto& r:grp) merged+=r.num_records;
                nruns.push_back({ooff,merged}); ooff+=merged*RECORD_SIZE;
            }
        }
        int out_runs=(int)nruns.size();
        double bw=(2.0*total_bytes)/(pass_ms*1e6);
        cum_traffic += 2.0*total_bytes;
        stats.push_back({pass,cur,out_runs,pass_blk,pass_ms,bw,
                         100.0*bw/peak_bw, 100.0*bw/copy_bw,
                         num_records/(pass_ms*1e3)});
        runs=nruns; std::swap(d_src,d_dst);
    }

    // ── Print per-pass results ──
    for (auto& s : stats)
        printf("  Pass %d: %d->%d runs | %.2f ms | %.1f GB/s "
               "(%.2f%% peak, %.1f%% copy) | %d blocks | %.0fM rec/s\n",
               s.pass,s.in_runs,s.out_runs,s.ms,s.bw_gbs,
               s.bw_pct,s.copy_pct,s.blocks,s.mrec_per_sec);

    // ── Summary ──
    float merge_ms=0; for(auto& s:stats) merge_ms+=s.ms;
    float tot_ms = rg_ms+merge_ms;
    double data_gb = total_bytes/1e9;
    double theo_merge = data_gb*2.0*ceil(log((double)num_runs)/log((double)fanin));
    double theo_total = theo_merge + 2.0*data_gb;
    double act_total  = cum_traffic/1e9;

    printf("\n%s\n  Summary\n%s\n",
           "═══════════════════════════════════════════════════",
           "═══════════════════════════════════════════════════");
    printf("  Total time:         %.2f ms (gen: %.2f + merge: %.2f)\n",
           tot_ms, rg_ms, merge_ms);
    printf("  Total HBM traffic:  %.2f GB (theoretical min: %.2f GB)\n",
           act_total, theo_total);
    printf("  Traffic efficiency: %.1f%%\n", 100.0*theo_total/act_total);
    printf("  Overall throughput: %.2f GB/s\n", total_bytes/(tot_ms*1e6));
    printf("  Records/sec:        %.1fM\n", num_records/(tot_ms*1e3));
    printf("  Copy ceiling:       %.1f GB/s\n", copy_bw);

    double avg_bw=0;
    if (!stats.empty()) { for(auto& s:stats) avg_bw+=s.bw_gbs; avg_bw/=stats.size(); }
    double ratio = copy_bw / std::max(avg_bw, 0.001);
    printf("  Avg merge BW:       %.1f GB/s\n", avg_bw);
    printf("  Sort is %.0fx below copy ceiling", ratio);
    if (ratio>10) printf(" -> COMPUTE / SMEM BOUND\n");
    else if (ratio>3) printf(" -> PARTIALLY BANDWIDTH LIMITED\n");
    else printf(" -> NEAR BANDWIDTH CEILING\n");
    printf("%s\n\n","═══════════════════════════════════════════════════");

    CUDA_CHECK(cudaFree(d_data)); CUDA_CHECK(cudaFree(d_sorted));
    CUDA_CHECK(cudaFree(d_mbuf));
    return 0;
}
