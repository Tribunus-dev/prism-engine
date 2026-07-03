// tile640_oracle.cpp — verifies (1) the base-3 tile640 (1.6-bit) pack/unpack +
// fp32 scaled dequant that the rewritten Metal kernel implements, and
// (2) quantifies the reconstruction error of the current absmax quantizer vs a
// BitNet-b1.58 absmean quantizer, to substantiate the "is the math sound?" review.
//
// tile640 layout (from repack_ternary_to_swizzled_u8 / decode_ternary_u32):
//   per row: nt = ceil(cols/640) tiles; per (tile t, lane 0..31) one u32 holds
//   20 base-3 trits; col = t*640 + lane*20 + vi; trit vi = (u32 / 3^vi) % 3;
//   digit 0 -> 0, 1 -> +1, 2 -> -1. Block scale is per 256 flattened elements.
//
// Build & run:  g++ -O2 -std=c++17 tile640_oracle.cpp -o t640 && ./t640
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
using namespace std;

static const uint32_t POW3[21] = {
    1u,3u,9u,27u,81u,243u,729u,2187u,6561u,19683u,59049u,177147u,531441u,
    1594323u,4782969u,14348907u,43046721u,129140163u,387420489u,1162261467u,3486784401u};

// digit -> signed ternary value: 0->0, 1->+1, 2->-1
static inline float trit_value(uint32_t d){ return d==1u?1.0f:(d==2u?-1.0f:0.0f); }
static inline uint32_t sign_to_digit(int s){ return s==1?1u:(s==-1?2u:0u); }

// ── two quantizers, per 256-block ───────────────────────────────────────────
// current engine: scale = absmax, q = round(w/scale) in {-1,0,1}
static int quant_absmax(float w, float scale){ int s=(int)lroundf(w/scale); return s<-1?-1:(s>1?1:s); }
// BitNet b1.58: scale = mean(|w|), q = round(w/scale) clamped
static int quant_absmean(float w, float scale){ int s=(int)lroundf(w/scale); return s<-1?-1:(s>1?1:s); }

int main(){
    const uint32_t rows=256, cols=1280;      // cols multiple of 640 (2 tiles)
    const uint32_t nt = (cols+639)/640;
    mt19937 rng(7);
    normal_distribution<float> nd(0.f,0.02f); // realistic weight scale

    vector<float> W(rows*cols); for(auto&v:W) v=nd(rng);
    vector<float> x(cols); { uniform_real_distribution<float> u(-1,1); for(auto&v:x) v=u(rng);}

    // Per-256-block scales (flattened row-major), for both schemes.
    const uint32_t nblk=(rows*cols+255)/256;
    vector<float> sc_absmax(nblk), sc_absmean(nblk);
    for(uint32_t b=0;b<nblk;++b){
        uint32_t st=b*256, n=min(256u, rows*cols-st);
        float mx=0.f, sum=0.f; for(uint32_t j=0;j<n;++j){float a=fabsf(W[st+j]); mx=fmaxf(mx,a); sum+=a;}
        sc_absmax[b]  = mx>1e-12f?mx:1.f;
        sc_absmean[b] = (sum/n)>1e-12f?(sum/n):1.f;   // b1.58
    }

    // Pack tile640 using the absmean (b1.58) trits — this is what the proper
    // kernel consumes. Store u32s in row/tile/lane order.
    vector<uint32_t> packed(rows*nt*32, 0);
    for(uint32_t r=0;r<rows;++r)
      for(uint32_t t=0;t<nt;++t)
        for(uint32_t lane=0;lane<32;++lane){
            uint32_t word=0;
            for(uint32_t vi=0;vi<20;++vi){
                uint32_t col=t*640+lane*20+vi; if(col>=cols) break;
                uint32_t flat=r*cols+col; uint32_t blk=flat/256;
                int s=quant_absmean(W[flat], sc_absmean[blk]);
                word += sign_to_digit(s)*POW3[vi];
            }
            packed[r*nt*32 + t*32 + lane]=word;
        }

    // Reference dequant+GEMV (direct) vs kernel-equivalent (unpack base-3).
    vector<double> y_ref(rows,0.0), y_ker(rows,0.0);
    for(uint32_t r=0;r<rows;++r){
        double acc=0.0;
        for(uint32_t c=0;c<cols;++c){ uint32_t blk=(r*cols+c)/256; int s=quant_absmean(W[r*cols+c],sc_absmean[blk]);
            acc += (double)x[c]*(double)(trit_value(sign_to_digit(s))*sc_absmean[blk]); }
        y_ref[r]=acc;
    }
    for(uint32_t r=0;r<rows;++r){
        float acc=0.f; // fp32 accumulation, as the rewritten kernel does
        for(uint32_t t=0;t<nt;++t) for(uint32_t lane=0;lane<32;++lane){
            uint32_t word=packed[r*nt*32+t*32+lane]; uint32_t rem=word;
            for(uint32_t vi=0;vi<20;++vi){ uint32_t col=t*640+lane*20+vi; uint32_t d=rem%3; rem/=3; if(col>=cols) continue;
                uint32_t blk=(r*cols+col)/256; acc += x[col]*(trit_value(d)*sc_absmean[blk]); }
        }
        y_ker[r]=acc;
    }
    double maxd=0; for(uint32_t r=0;r<rows;++r) maxd=fmax(maxd,fabs(y_ref[r]-y_ker[r]));
    printf("[unpack] tile640 base-3 + fp32 scaled dequant: max|ref - kernel| = %.3e  %s\n",
           maxd, maxd<1e-3?"PASS":"FAIL");

    // Reconstruction error: absmax vs absmean (b1.58). Lower is better.
    double se_mx=0, se_mn=0, e0_mx=0, e0_mn=0, denom=0;
    for(uint32_t r=0;r<rows;++r) for(uint32_t c=0;c<cols;++c){
        uint32_t flat=r*cols+c, blk=flat/256; float w=W[flat];
        float rmx=trit_value(sign_to_digit(quant_absmax(w,sc_absmax[blk])))*sc_absmax[blk];
        float rmn=trit_value(sign_to_digit(quant_absmean(w,sc_absmean[blk])))*sc_absmean[blk];
        se_mx+=(w-rmx)*(w-rmx); se_mn+=(w-rmn)*(w-rmn); denom+=w*w;
        if(rmx==0.f) e0_mx++; if(rmn==0.f) e0_mn++;
    }
    double N=rows*cols;
    printf("[quantizer] relative L2 error  absmax=%.3f  absmean(b1.58)=%.3f  (lower=better)\n",
           sqrt(se_mx/denom), sqrt(se_mn/denom));
    printf("[quantizer] fraction zeroed    absmax=%.1f%%  absmean(b1.58)=%.1f%%\n",
           100.0*e0_mx/N, 100.0*e0_mn/N);
    return maxd<1e-3?0:1;
}
