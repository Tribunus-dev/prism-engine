#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGUF magic bytes 'GGUF'
#define GGUF_MAGIC 0x46554747 // Little-endian representation of "GGUF"

// GGUF value types
typedef enum gguf_value_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
} gguf_value_type_t;

// Opaque struct representing a GGUF file
typedef struct gguf_file gguf_file_t;

// Exposes struct fields
struct gguf_file {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    // Internal fields will be added in src/gguf.c or we can keep it here
    void* internal_data; 
};

/**
 * Reads and validates a GGUF file header.
 * @param path The file path to read.
 * @return A pointer to a new gguf_file_t on success, or NULL on error.
 */
gguf_file_t* gguf_file_read(const char* path);

/**
 * Frees a GGUF file structure.
 */
void gguf_file_free(gguf_file_t* f);

/**
 * Reads a metadata KV pair at the specified index.
 * @param f The GGUF file.
 * @param index The index of the metadata KV pair.
 * @param key Will point to the key string.
 * @param type Will point to the value type.
 * @param value Will point to the value payload (could be string struct, array struct, or primitive).
 * @return true on success, false on error.
 */
bool gguf_file_metadata(gguf_file_t* f, uint32_t index, const char** key, gguf_value_type_t* type, void** value);

// GGUF string structure
typedef struct gguf_string {
    uint64_t len;
    char* data;
} gguf_string_t;

// GGUF array structure
typedef struct gguf_array {
    gguf_value_type_t type;
    uint64_t len;
    void* data;
} gguf_array_t;

// Session 3 (LLaMA) GGUF load API
typedef gguf_value_type_t gguf_type;

typedef struct {
    char magic[4];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
} gguf_header;

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t shape[4];
    gguf_type type;
    uint64_t offset;
    void *data;
} gguf_tensor_info;

typedef struct {
    char *name;
    gguf_type type;
    void *value;
} gguf_kv_info;

typedef struct {
    gguf_header header;
    gguf_kv_info *kvs;
    gguf_tensor_info *tensors;
    void *mmap_addr;
    size_t mmap_size;
    int fd;
    uint64_t alignment;
    uint64_t data_offset;
} gguf_context;

gguf_context* gguf_load(const char *filename);
void gguf_free(gguf_context *ctx);
gguf_tensor_info* gguf_get_tensor(gguf_context *ctx, const char *name);


#ifdef __cplusplus
}
#endif

#endif // GGUF_H
