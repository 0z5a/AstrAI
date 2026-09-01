/*
Pure-C test — uses shared dispatcher.  Combines the decode (split-KV) and
prefill (split-Q) correctness checks + benchmarks into one binary.
nvcc -I csrc/kernels -arch=sm_89 -O3 \
    --use_fast_math --ptxas-options=-O3 --extra-device-vectorization \
    -Xcompiler -fopenmp csrc/tests/attn_test.cu -o test && ./test
*/

#include "test_utils.cuh"
#include "attention/dispatchers.cuh"

using namespace astrai::attention;

struct DecodeDispatch { AttentionParams<bf16>& p; template<int H> void operator()() { dispatch_decode<H>(p, 0); } };
struct PrefillDispatch { AttentionParams<bf16>& p; template<int H> void operator()() { dispatch_prefill<H>(p, 0); } };

// Split-K scratch (torch-free)
struct DecodeScratch {
    float* o_part = nullptr;
    float* ml_part = nullptr;
};

static void setup_scratch(AttentionParams<bf16>& p, DecodeScratch& sc) {
    int max_splits = 32;
    cudaMalloc(&sc.o_part, (size_t)p.batch * p.q_head * max_splits * p.head_dim * sizeof(float));
    cudaMalloc(&sc.ml_part, (size_t)p.batch * p.q_head * max_splits * 2 * sizeof(float));
}

static void free_scratch(DecodeScratch& sc) {
    cudaFree(sc.o_part); cudaFree(sc.ml_part);
}

// ======================================================================
// DECODE
// ======================================================================

static int run_decode_test(int B, int Hq, int Hk, int sl, int D, int causal) {
    size_t nQ = B*Hq*1*D, nKV = B*Hk*sl*D;
    float *hQ=new float[nQ], *hK=new float[nKV], *hV=new float[nKV];
    for (size_t i=0;i<nQ;i++) hQ[i]=randf();
    for (size_t i=0;i<nKV;i++){hK[i]=randf();hV[i]=randf();}

    bool* hMask=new bool[B*sl];
    for (int i=0;i<B*sl;i++) hMask[i]=true;

    bf16 *dQ,*dK,*dV,*dO,*tmp;
    bool* dMask;
    cudaMalloc(&dQ,nQ*2); cudaMalloc(&dK,nKV*2);
    cudaMalloc(&dV,nKV*2); cudaMalloc(&dO,nQ*2);
    cudaMalloc(&dMask,B*sl);

    tmp=new bf16[max(nQ,nKV)];
    for (size_t i=0;i<nQ;i++) tmp[i]=f2bf(hQ[i]);
    cudaMemcpy(dQ,tmp,nQ*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hK[i]);
    cudaMemcpy(dK,tmp,nKV*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hV[i]);
    cudaMemcpy(dV,tmp,nKV*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dMask,hMask,B*sl,cudaMemcpyHostToDevice);

    AttentionParams<bf16> p = {};
    p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=1; p.kv_len=sl; p.head_dim=D;
    p.use_mask=0; p.causal_offset=causal?0:-1;
    p.scale=1.0f/sqrtf((float)D);
    set_default_strides(p);
    p.q_ptr=dQ; p.k_ptr=dK; p.v_ptr=dV; p.mask=nullptr; p.o_ptr=dO;

    DecodeScratch sc;
    setup_scratch(p, sc);
    p.o_part = sc.o_part; p.ml_part = sc.ml_part;

    double t0=now_ms();
    dispatch_by_head_dim(D, DecodeDispatch{p});
    cudaDeviceSynchronize();
    (void)t0;
    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

    bf16* hOut=new bf16[nQ];
    cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

    float* ref=new float[nQ];
    cpu_attention_ref(hQ, hK, hV, hMask, ref, B, Hq, Hk, 1, sl, D, causal ? 0 : -1);

    float max_abs_err=0, max_rel_err=0;
    for (size_t i=0;i<nQ;i++){
        float err=fabsf(bf2f(hOut[i])-ref[i]);
        if(err>max_abs_err) max_abs_err=err;
        float rel=err/fmaxf(fabsf(ref[i]), 1e-4f);
        if(rel>max_rel_err) max_rel_err=rel;
    }
    const float atol=0.01f, rtol=0.01f;
    bool pass=true;
    for (size_t i=0;i<nQ;i++){
        float err=fabsf(bf2f(hOut[i])-ref[i]);
        if (err > atol + rtol * fabsf(ref[i])) { pass=false; break; }
    }
    char cfg[64];
    snprintf(cfg, sizeof(cfg), "B=%2d Hq=%2d Hk=%d seq=%4d D=%3d causal=%d",
             B, Hq, Hk, sl, D, causal);
    print_test_row(cfg, max_abs_err, max_rel_err, pass);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dMask);
    free_scratch(sc);
    delete[]hQ;delete[]hK;delete[]hV;delete[]hMask;delete[]hOut;delete[]ref;delete[]tmp;

    return pass ? 0 : 1;
}

static void bench_decode() {
    const int cfgs[][5] = {
        {1, 32, 4, 512, 128},
        {1, 32, 4, 1024, 128},
        {1, 32, 4, 2048, 128},
        {1, 32, 4, 4096, 128},
        {16, 32, 4, 2048, 128},
        {32, 32, 4, 1024, 128},
    };
    const int WARMUP = 3, ITERS = 10;
    printf("\n===== DECODE BENCH (warmup=%d iters=%d) =====\n", WARMUP, ITERS);
    print_bench_header();

    int n = sizeof(cfgs) / sizeof(cfgs[0]);
    for (int ci = 0; ci < n; ci++) {
        int B = cfgs[ci][0], Hq = cfgs[ci][1], Hk = cfgs[ci][2];
        int sl = cfgs[ci][3], D = cfgs[ci][4];
        size_t nQ = (size_t)B * Hq * D;
        size_t nKV = (size_t)B * Hk * sl * D;

        bf16 *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, nQ*2); cudaMalloc(&dK, nKV*2);
        cudaMalloc(&dV, nKV*2); cudaMalloc(&dO, nQ*2);
        size_t big = nQ > nKV ? nQ : nKV; bf16* tmp = new bf16[big];
        for (size_t i = 0; i < nQ; i++)  tmp[i] = f2bf(randf());
        cudaMemcpy(dQ, tmp, nQ*2, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < nKV; i++) tmp[i] = f2bf(randf());
        cudaMemcpy(dK, tmp, nKV*2, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < nKV; i++) tmp[i] = f2bf(randf());
        cudaMemcpy(dV, tmp, nKV*2, cudaMemcpyHostToDevice);
        delete[] tmp;

        AttentionParams<bf16> p = {};
        p.batch = B; p.q_head = Hq; p.kv_head = Hk; p.q_len = 1; p.kv_len = sl;
        p.head_dim = D; p.use_mask = 0; p.causal_offset = -1;
        p.scale = 1.0f / sqrtf((float)D);
        set_default_strides(p);
        p.q_ptr = dQ; p.k_ptr = dK; p.v_ptr = dV; p.mask = nullptr; p.o_ptr = dO;

        DecodeScratch sc;
        setup_scratch(p, sc);
        p.o_part = sc.o_part; p.ml_part = sc.ml_part;

        auto launch = [&]() { dispatch_by_head_dim(D, DecodeDispatch{p}); };
        double flops = 4.0 * B * Hq * (double)sl * D;
        BenchResult r = bench_kernel(launch, WARMUP, ITERS, flops);

        char cfg[64];
        snprintf(cfg, sizeof(cfg),
                 "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d",
                 B, Hq, Hk, 1, sl, D, 0);
        print_bench_row(cfg, r);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        free_scratch(sc);
    }
}

// ======================================================================
// PREFILL
// ======================================================================

static int run_prefill_test(int B, int Hq, int Hk, int ql, int kl, int D, int causal) {
    size_t nQ = B*Hq*ql*D, nKV = B*Hk*kl*D;
    float *hQ=new float[nQ], *hK=new float[nKV], *hV=new float[nKV];
    for (size_t i=0;i<nQ;i++) hQ[i]=randf();
    for (size_t i=0;i<nKV;i++){hK[i]=randf();hV[i]=randf();}

    bf16 *dQ,*dK,*dV,*dO,*tmp;
    cudaMalloc(&dQ,nQ*2); cudaMalloc(&dK,nKV*2);
    cudaMalloc(&dV,nKV*2); cudaMalloc(&dO,nQ*2);
    tmp=new bf16[max(nQ,nKV)];
    for (size_t i=0;i<nQ;i++) tmp[i]=f2bf(hQ[i]);
    cudaMemcpy(dQ,tmp,nQ*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hK[i]);
    cudaMemcpy(dK,tmp,nKV*2,cudaMemcpyHostToDevice);
    for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(hV[i]);
    cudaMemcpy(dV,tmp,nKV*2,cudaMemcpyHostToDevice);

    AttentionParams<bf16> p = {};
    p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=ql; p.kv_len=kl; p.head_dim=D;
    p.use_mask=0; p.causal_offset=causal?0:-1;
    set_default_strides(p);
    p.scale=1.0f/sqrtf((float)D);
    p.q_ptr=dQ; p.k_ptr=dK; p.v_ptr=dV; p.mask=nullptr; p.o_ptr=dO;

    double t0=now_ms();
    dispatch_by_head_dim(D, PrefillDispatch{p});
    cudaDeviceSynchronize();
    (void)t0;
    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return 1;}

    bf16* hOut=new bf16[nQ];
    cudaMemcpy(hOut,dO,nQ*2,cudaMemcpyDeviceToHost);

    float* ref=new float[nQ];
    cpu_attention_ref(hQ, hK, hV, nullptr, ref, B, Hq, Hk, ql, kl, D, causal ? 0 : -1);

    float max_abs_err=0, max_rel_err=0;
    for (size_t i=0;i<nQ;i++) {
        float err=fabsf(bf2f(hOut[i])-ref[i]);
        if(err>max_abs_err) max_abs_err=err;
        float rel=err/fmaxf(fabsf(ref[i]), 1e-4f);
        if(rel>max_rel_err) max_rel_err=rel;
    }
    const float atol=0.01f, rtol=0.01f;
    bool pass=true;
    for (size_t i=0;i<nQ;i++) {
        float err=fabsf(bf2f(hOut[i])-ref[i]);
        if (err > atol + rtol * fabsf(ref[i])) { pass=false; break; }
    }
    char cfg[64];
    snprintf(cfg, sizeof(cfg), "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d",
             B, Hq, Hk, ql, kl, D, causal);
    print_test_row(cfg, max_abs_err, max_rel_err, pass);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    delete[]hQ;delete[]hK;delete[]hV;delete[]hOut;delete[]ref;delete[]tmp;

    return pass ? 0 : 1;
}

static void bench_prefill() {
    const int cfgs[][7] = {
        {1,32,4,1024,1024,32,0},
        {1,32,4,1024,1024,32,1},
        {1,32,4,4096,4096,32,1},
        {1,32,4,1024,1024,64,0},
        {1,32,4,1024,1024,64,1},
        {1,32,4,4096,4096,64,1},
        {1,32,4,512,512,128,0},
        {1,32,4,1024,1024,128,0},
        {1,32,4,2048,2048,128,0},
        {1,32,4,2048,2048,128,1},
        {4,32,4,2048,2048,128,1},
        {1,32,4,4096,4096,128,1},
    };
    int n = sizeof(cfgs)/sizeof(cfgs[0]);
    const int WARMUP = 3, ITERS = 10;
    printf("\n===== PREFILL BENCH (warmup=%d iters=%d) =====\n", WARMUP, ITERS);
    print_bench_header();

    for (int ci = 0; ci < n; ci++) {
        int B=cfgs[ci][0], Hq=cfgs[ci][1], Hk=cfgs[ci][2];
        int ql=cfgs[ci][3], kl=cfgs[ci][4], D=cfgs[ci][5], causal=cfgs[ci][6];
        size_t nQ=(size_t)B*Hq*ql*D, nKV=(size_t)B*Hk*kl*D;

        bf16 *dQ,*dK,*dV,*dO,*tmp;
        cudaMalloc(&dQ,nQ*2); cudaMalloc(&dK,nKV*2);
        cudaMalloc(&dV,nKV*2); cudaMalloc(&dO,nQ*2);
        size_t big = nQ>nKV?nQ:nKV; tmp=new bf16[big];
        for (size_t i=0;i<nQ;i++)  tmp[i]=f2bf(randf());
        cudaMemcpy(dQ,tmp,nQ*2,cudaMemcpyHostToDevice);
        for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(randf());
        cudaMemcpy(dK,tmp,nKV*2,cudaMemcpyHostToDevice);
        for (size_t i=0;i<nKV;i++) tmp[i]=f2bf(randf());
        cudaMemcpy(dV,tmp,nKV*2,cudaMemcpyHostToDevice);

        AttentionParams<bf16> p = {};
        p.batch=B; p.q_head=Hq; p.kv_head=Hk; p.q_len=ql; p.kv_len=kl; p.head_dim=D;
        p.use_mask=0; p.causal_offset=causal?0:-1;
        set_default_strides(p);
        p.scale=1.0f/sqrtf((float)D);
        p.q_ptr=dQ; p.k_ptr=dK; p.v_ptr=dV; p.mask=nullptr; p.o_ptr=dO;

        auto launch = [&]() { dispatch_by_head_dim(D, PrefillDispatch{p}); };
        for (int i=0;i<WARMUP;i++) launch();
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if (err!=cudaSuccess){printf("CUDA err: %s\n",cudaGetErrorString(err));return;}

        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i=0;i<ITERS;i++) launch();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=ITERS;

        double flops = 4.0*B*Hq*(double)ql*kl*D;
        if (causal) flops *= 0.5;
        double tflops = flops/(ms*1e-3)/1e12;
        BenchResult r{ms, tflops};

        char cfg[64];
        snprintf(cfg, sizeof(cfg),
                 "B=%2d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d causal=%d",
                 B,Hq,Hk,ql,kl,D,causal);
        print_bench_row(cfg, r);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
        delete[]tmp; cudaEventDestroy(s); cudaEventDestroy(e);
    }
}

// ======================================================================
// MAIN
// ======================================================================

int main() {
    int fail = 0;

    // ---- DECODE ----
    {
        const int configs[][6] = {
            {1, 2, 1, 64, 32, 0},
            {1, 32, 4, 512, 128, 0},
            {1, 32, 4, 1024, 128, 0},
            {1, 32, 4, 512, 128, 1},
        };
        int n_cfgs = sizeof(configs) / sizeof(configs[0]);
        printf("=== DECODE TESTS ===\n");
        print_test_header();
        for (int ci = 0; ci < n_cfgs; ci++) {
            int B = configs[ci][0], Hq = configs[ci][1], Hk = configs[ci][2];
            int sl = configs[ci][3], D = configs[ci][4], causal = configs[ci][5];
            fail += run_decode_test(B, Hq, Hk, sl, D, causal);
            if (fail) break;
        }
        if (fail) { printf("FAILED decode tests\n"); return fail; }
        bench_decode();
    }

    // ---- PREFILL ----
    {
        const int configs[][7] = {
            {1,2,1,64,128,32,0},     // scalar fallback D=32
            {1,4,2,256,256,32,1},     // causal D=32 dispatch
            {1,2,1,64,128,64,0},     // tiny: B,Hq,Hk,q,kv,D,causal
            {1,4,2,256,256,64,1},     // causal D=64 dispatch
            {1,32,4,512,512,128,0},  // standard
            {1,32,4,128,256,128,0},  // medium
            {1,4,2,256,256,128,1},   // causal
        };
        int n_configs = sizeof(configs) / sizeof(configs[0]);
        printf("\n=== PREFILL TESTS ===\n");
        print_test_header();
        for (int ci = 0; ci < n_configs; ci++) {
            int B=configs[ci][0], Hq=configs[ci][1], Hk=configs[ci][2];
            int ql=configs[ci][3], kl=configs[ci][4], D=configs[ci][5];
            int causal=configs[ci][6];
            fail += run_prefill_test(B, Hq, Hk, ql, kl, D, causal);
            if (fail) break;
        }
        if (fail) { printf("FAILED prefill tests\n"); return fail; }
        bench_prefill();
    }

    printf("\nAll tests passed!\n");
    return 0;
}
