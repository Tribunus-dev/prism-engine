// Xcode 26.5+ Metal SDK provides bfloat16 math natively AND macOS 26's Metal
// may define `half` math functions that collide with the generic macro.
// Guard both conditions: bfloat must be a real extension AND version < 310000.
#if __has_extension(metal_bfloat) && __METAL_VERSION__ < 310000
