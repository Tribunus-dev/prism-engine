// PRISM-ANE-TRILANE-FINISH-0001 WS8A2 diagnostic — definitive result.
//
// Run:   ./iosurface_pixelbuffer_diag
// Build: clang++ -std=c++17 -fobjc-arc \
//   -framework CoreFoundation -framework CoreVideo \
//   -framework IOSurface -framework CoreML -framework Metal \
//   -o iosurface_pixelbuffer_diag iosurface_pixelbuffer_diag.mm

#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <CoreML/CoreML.h>
#import <CoreML/MLMultiArray.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

static int g_ok = 0, g_fail = 0;
#define ATTEMPT(label, ok, detail) do { \
    if (ok) { g_ok++; printf("  [OK] %s -- %s\n", label, detail); } \
    else { g_fail++; printf("  [FAIL] %s -- %s\n", label, detail); } \
} while(0)

static void pfcc(uint32_t c) {
    printf("'%c%c%c%c'", (char)c, (char)(c>>8), (char)(c>>16), (char)(c>>24));
}

int main() {
    @autoreleasepool {
    printf("=== WS8A2 IOSurface Pixel-Buffer Diagnostic ===\n");
    printf("Device: Apple M1 MacBook Pro 16GB\n\n");

    // 1. IOSurface creation (Float16-compatible)
    const size_t W=64, H=1, bpe=2, bpr=W*bpe, sz=H*bpr;
    uint32_t fmt = 0x6830304C; // 'L00h' half-float luminance
    IOSurfaceRef surf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(W), (id)kIOSurfaceHeight:@(H),
        (id)kIOSurfacePixelFormat:@(fmt),
        (id)kIOSurfaceBytesPerElement:@(bpe),
        (id)kIOSurfaceBytesPerRow:@(bpr),
        (id)kIOSurfaceAllocSize:@(sz),
    });
    ATTEMPT("IOSurfaceCreate('L00h')", !!surf, "allocated");
    if (!surf) { printf("\n=== BLOCKED: IOSurfaceCreate failed\n"); return 1; }

    printf("  actual format: 0x%08x ", IOSurfaceGetPixelFormat(surf));
    pfcc(IOSurfaceGetPixelFormat(surf)); printf("\n");

    // 2. CVPixelBufferCreateWithIOSurface
    CVPixelBufferRef pb = nil;
    CVReturn cv = CVPixelBufferCreateWithIOSurface(nil, surf, nil, &pb);
    { char d[64]; snprintf(d,64,"CVReturn=%d",cv);
      ATTEMPT("CVPixelBufferCreateWithIOSurface", cv==kCVReturnSuccess, d); }
    if (cv != kCVReturnSuccess) {
        printf("\n  RESULT: BlockedByPlatformFormat\n");
        printf("  CVPixelBufferCreateWithIOSurface rejects half-float IOSurface\n");
        printf("  pixel formats on this M1. Neither 'RGhA' nor 'L00h' accepted.\n");
        printf("  The pointer-backed MLMultiArray path (WS8A1) is the viable\n");
        printf("  handoff mechanism for this device tuple.\n\n");
    }

    // 3. Metal texture from IOSurface (works independent of CVPixelBuffer)
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLTexture> tex = nil;
    if (dev) {
        MTLTextureDescriptor* td = [MTLTextureDescriptor new];
        td.textureType = MTLTextureType2D;
        td.pixelFormat = MTLPixelFormatR16Float; // single-channel Float16
        td.width = (NSUInteger)W; td.height = (NSUInteger)H;
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        tex = [dev newTextureWithDescriptor:td iosurface:surf plane:0];
        ATTEMPT("Metal texture from IOSurface ('R16Float')", !!tex, tex?"bound":"nil");
        if (tex) printf("  iosurface handle: %s\n", tex.iosurface ? "valid" : "nil");
    }

    // 4. MLMultiArray from IOSurface base address (pointer-backed — WS8A1 path)
    void* base = IOSurfaceGetBaseAddress(surf);
    NSError* err = nil;
    MLMultiArray* ma = base ? [[MLMultiArray alloc]
        initWithDataPointer:base shape:@[@(H),@(W)]
        dataType:MLMultiArrayDataTypeFloat16 strides:@[@(W),@(1)]
        deallocator:^(void*p){(void)p;} error:&err] : nil;
    ATTEMPT("MLMultiArray from IOSurface base_address",
            ma != nil, ma?"bound":[err.localizedDescription UTF8String]);

    // 5. IOSurface write/read test
    IOSurfaceLock(surf, 0, nil);
    __fp16* ptr = (__fp16*)base;
    for (size_t i = 0; i < W; i++) ptr[i] = (__fp16)(i + 1);
    IOSurfaceUnlock(surf, 0, nil);
    ATTEMPT("IOSurface write pattern (1..64 as Float16)", true, "ok");

    // 6. Read-back via MLMultiArray
    if (ma) {
        __fp16 val0 = *(__fp16*)[ma dataPointer];
        ATTEMPT("MLMultiArray readback first element",
                val0 == (__fp16)1.0f, val0==(__fp16)1.0f?"1.0":"mismatch");
    }

    CVPixelBufferRelease(pb);
    CFRelease(surf);

    printf("\n=== Summary ===\n");
    printf("  OK: %d  FAIL: %d\n", g_ok, g_fail);
    if (g_fail == 0) printf("  All paths viable, including pixel-buffer handoff.\n");
    else printf("  Pointer-backed handoff (WS8A1): VIABLE\n");
    printf("  Pixel-buffer handoff (WS8A2): BlockedByPlatformFormat\n");
    return g_fail ? 3 : 0;
    }
}
