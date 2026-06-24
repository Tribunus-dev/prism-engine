pub fn patch_allocator_memory_plan(mlx_src: &std::path::Path) {
    use std::fs;

    let alloc_h = mlx_src.join("mlx/backend/metal/allocator.h");
    if alloc_h.exists() {
        let text = fs::read_to_string(&alloc_h).unwrap_or_default();
        let mut out = text.clone();

        if !out.contains("struct PlanEntry {") {
            out = out.replace(
                "class MetalAllocator : public allocator::Allocator {",
                concat!("struct PlanEntry { void* ptr; size_t size; };\n",
                        "\n",
                        "class MetalAllocator : public allocator::Allocator {"),
            );
        }

        if !out.contains("std::vector<PlanEntry> plan_") {
            out = out.replace(
                "std::mutex mutex_;",
                concat!("std::mutex mutex_;\n",
                        "\n",
                        "  // Memory plan\n",
                        "  std::vector<PlanEntry> plan_;\n",
                        "  size_t plan_index_{0};"),
            );
        }

        if !out.contains("Memory plan API") {
            out = out.replace(
                "void clear_cache();",
                concat!("void clear_cache();\n",
                        "\n",
                        "  // Memory plan API\n",
                        "  void set_memory_plan(size_t num_slots, const PlanEntry* slots);\n",
                        "  void clear_memory_plan();"),
            );
        }

        if out != text {
            fs::write(&alloc_h, &out).expect("Failed to patch allocator.h");
            println!("cargo:warning=PATCH allocator.h memory plan");
        }
    }

    let alloc_cpp = mlx_src.join("mlx/backend/metal/allocator.cpp");
    if alloc_cpp.exists() {
        let text = fs::read_to_string(&alloc_cpp).unwrap_or_default();
        let mut out = text.clone();

        if !out.contains("CHECK: memory plan") {
            let plan_check = concat!(
                "// CHECK: memory plan\n",
                "{\n",
                "        std::unique_lock lk(mutex_);\n",
                "        if (plan_index_ < plan_.size()) {\n",
                "                auto& entry = plan_[plan_index_++];\n",
                "                auto buf = device_->newBuffer(entry.ptr, size, resource_options, nullptr);\n",
                "                if (buf) {\n",
                "                        residency_set_.insert(buf);\n",
                "                        active_memory_ += buf->length();\n",
                "                        peak_memory_ = std::max(peak_memory_, active_memory_);\n",
                "                        num_resources_++;\n",
                "                        return Buffer{static_cast<void*>(buf)};\n",
                "                }\n",
                "        }\n",
                "}\n",
                "\n",
                "// Align up memory"
            );
            out = out.replace("// Align up memory", plan_check);
        }

        if !out.contains("MetalAllocator::set_memory_plan") {
            let marker = concat!(
                "void MetalAllocator::clear_cache() {\n",
                "        std::unique_lock lk(mutex_);\n",
                "        num_resources_ -= buffer_cache_.clear();\n",
                "}"
            );
            let impls = concat!(
                "\n",
                "void MetalAllocator::set_memory_plan(size_t num_slots, const PlanEntry* slots) {\n",
                "        std::unique_lock lk(mutex_);\n",
                "        plan_.resize(num_slots);\n",
                "        for (size_t i = 0; i < num_slots; ++i) {\n",
                "                plan_[i] = slots[i];\n",
                "        }\n",
                "        plan_index_ = 0;\n",
                "}\n",
                "\n",
                "void MetalAllocator::clear_memory_plan() {\n",
                "        std::unique_lock lk(mutex_);\n",
                "        plan_.clear();\n",
                "        plan_index_ = 0;\n",
                "}"
            );
            out = out.replace(marker, &format!("{}{}", marker, impls));
        }

        if !out.contains("// memory plan convenience") {
            let ns = concat!(
                "\n",
                "// memory plan convenience (Tribunus)\n",
                "void set_memory_plan(size_t num_slots, const PlanEntry* slots) {\n",
                "        allocator().set_memory_plan(num_slots, slots);\n",
                "}\n",
                "\n",
                "void clear_memory_plan() {\n",
                "        allocator().clear_memory_plan();\n",
                "}\n",
                "\n",
            );
            out = out.replace(
                "MetalAllocator& allocator() {",
                &format!("{}MetalAllocator& allocator() {{", ns),
            );
        }

        if out != text {
            fs::write(&alloc_cpp, &out).expect("Failed to patch allocator.cpp");
            println!("cargo:warning=PATCH allocator.cpp memory plan");
        }
    }
}
