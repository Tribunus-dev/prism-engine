# CMake patch script for Memory Plan support in MetalAllocator.
# Called from FetchContent_Declare PATCH_COMMAND with MLX_SRC defined.
#
# Modifies:
#   1. allocator.h — adds PlanEntry struct and plan fields to MetalAllocator
#   2. allocator.cpp — adds plan dispatch in malloc() and set/clear implementations
#
# The memory plan allows the executor to pre-arrange IOSurface-backed MTLBuffers.
# When malloc() is called with a plan active, it wraps the next plan entry's pointer
# as a zero-copy MTLBuffer via newBuffer(ptr, size, options, nullptr) instead of
# heap/cache allocation. The residency set is updated so Metal tracks the buffer.

# --- Patch allocator.h ---
set(ALLOC_H "${MLX_SRC}/mlx/backend/metal/allocator.h")
file(READ "${ALLOC_H}" ALLOC_H_CONTENT)

# Guard: check if PlanEntry struct already exists (idempotent)
string(FIND "${ALLOC_H_CONTENT}" "struct PlanEntry" PLANENTRY_POS)
if(PLANENTRY_POS EQUAL -1)
  # Insert PlanEntry struct before MetalAllocator class declaration
  string(REPLACE
    "class MetalAllocator : public allocator::Allocator {"
    "struct PlanEntry {\n  void* ptr;\n  size_t size;\n};\n\nclass MetalAllocator : public allocator::Allocator {"
    ALLOC_H_CONTENT "${ALLOC_H_CONTENT}")
  message(STATUS "  allocator.h: added PlanEntry struct")
endif()

# Guard: check if plan_ field already exists
string(FIND "${ALLOC_H_CONTENT}" "std::vector<PlanEntry> plan_" PLANFIELD_POS)
if(PLANFIELD_POS EQUAL -1)
  # Add plan fields after the mutex declaration
  string(REPLACE
    "  std::mutex mutex_;"
    "  std::mutex mutex_;\n\n  // Memory plan (Tribunus)\n  std::vector<PlanEntry> plan_;\n  size_t plan_index_{0};"
    ALLOC_H_CONTENT "${ALLOC_H_CONTENT}")
  message(STATUS "  allocator.h: added plan_/plan_index_ fields")
endif()

# Guard: check if set_memory_plan declaration already exists
string(FIND "${ALLOC_H_CONTENT}" "set_memory_plan" SETPLAN_POS)
if(SETPLAN_POS EQUAL -1)
  # Insert memory plan API declarations as public member functions
  # inside the MetalAllocator class.  Insert after clear_cache().
  string(REPLACE
    "  void clear_cache();"
    "  void clear_cache();\n\n  // Memory plan API (Tribunus)\n  void set_memory_plan(size_t num_slots, const PlanEntry* slots);\n  void clear_memory_plan();"
    ALLOC_H_CONTENT "${ALLOC_H_CONTENT}")
  message(STATUS "  allocator.h: added set_memory_plan/clear_memory_plan declarations")
endif()

if(NOT PLANENTRY_POS EQUAL -1 AND NOT PLANFIELD_POS EQUAL -1 AND NOT SETPLAN_POS EQUAL -1)
  message(STATUS "  allocator.h: already patched, skipping")
endif()

file(WRITE "${ALLOC_H}" "${ALLOC_H_CONTENT}")

# --- Patch allocator.cpp ---
set(ALLOC_CPP "${MLX_SRC}/mlx/backend/metal/allocator.cpp")
file(READ "${ALLOC_CPP}" ALLOC_CPP_CONTENT)

# Guard: check if memory plan dispatch already inserted in malloc
string(FIND "${ALLOC_CPP_CONTENT}" "// Check memory plan first" MALLOC_PLAN_POS)
if(MALLOC_PLAN_POS EQUAL -1)
  # Insert plan dispatch in malloc() — right before the alignment comment.
  # This runs BEFORE the cache check: if a plan entry is available, we wrap the
  # pre-allocated IOSurface pointer as a zero-copy MTLBuffer via newBuffer(ptr, size, ...).
  string(REPLACE
    "  // Align up memory"
    "  // Check memory plan first (Tribunus)\n  {\n    std::unique_lock lk(mutex_);\n    if (plan_index_ < plan_.size()) {\n      auto& entry = plan_[plan_index_++];\n      auto buf = device_->newBuffer(entry.ptr, size, resource_options, nullptr);\n      if (buf) {\n        residency_set_.insert(buf);\n        active_memory_ += buf->length();\n        peak_memory_ = std::max(peak_memory_, active_memory_);\n        num_resources_++;\n        return Buffer{static_cast<void*>(buf)};\n      }\n    }\n  }\n\n  // Align up memory"
    ALLOC_CPP_CONTENT "${ALLOC_CPP_CONTENT}")
  message(STATUS "  allocator.cpp: added plan dispatch in malloc()")
endif()

# Guard: check if set_memory_plan implementation already exists
string(FIND "${ALLOC_CPP_CONTENT}" "void MetalAllocator::set_memory_plan" SETPLAN_IMPL_POS)
if(SETPLAN_IMPL_POS EQUAL -1)
  # Add set_memory_plan and clear_memory_plan implementations after clear_cache().
  # Extend the clear_cache function definition to include these new functions.
  string(REPLACE
    "void MetalAllocator::clear_cache() {\n  std::unique_lock lk(mutex_);\n  num_resources_ -= buffer_cache_.clear();\n}"
    "void MetalAllocator::clear_cache() {\n  std::unique_lock lk(mutex_);\n  num_resources_ -= buffer_cache_.clear();\n}\n\nvoid MetalAllocator::set_memory_plan(size_t num_slots, const PlanEntry* slots) {\n  std::unique_lock lk(mutex_);\n  plan_.resize(num_slots);\n  for (size_t i = 0; i < num_slots; ++i) {\n    plan_[i] = slots[i];\n  }\n  plan_index_ = 0;\n}\n\nvoid MetalAllocator::clear_memory_plan() {\n  std::unique_lock lk(mutex_);\n  plan_.clear();\n  plan_index_ = 0;\n}"
    ALLOC_CPP_CONTENT "${ALLOC_CPP_CONTENT}")
  message(STATUS "  allocator.cpp: added set_memory_plan/clear_memory_plan implementations")
endif()

# Guard: check if namespace-level convenience functions already exist
string(FIND "${ALLOC_CPP_CONTENT}" "// namespace-level memory plan" NSPACE_PLAN_POS)
if(NSPACE_PLAN_POS EQUAL -1)
  # Add namespace-level convenience set_memory_plan / clear_memory_plan functions
  # that delegate to the allocator singleton. These sit just before allocator().
  string(REPLACE
    "MetalAllocator& allocator() {"
    "// namespace-level memory plan helpers (Tribunus)\nvoid set_memory_plan(size_t num_slots, const PlanEntry* slots) {\n  allocator().set_memory_plan(num_slots, slots);\n}\n\nvoid clear_memory_plan() {\n  allocator().clear_memory_plan();\n}\n\nMetalAllocator& allocator() {"
    ALLOC_CPP_CONTENT "${ALLOC_CPP_CONTENT}")
  message(STATUS "  allocator.cpp: added namespace-level set_memory_plan/clear_memory_plan")
endif()

file(WRITE "${ALLOC_CPP}" "${ALLOC_CPP_CONTENT}")

message(STATUS "MLX allocator patches complete")
