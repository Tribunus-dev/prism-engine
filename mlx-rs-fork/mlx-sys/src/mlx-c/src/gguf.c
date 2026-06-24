#include "../include/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct gguf_kv {
    gguf_string_t key;
    gguf_value_type_t type;
    void* value; // Points to allocated memory for the value
};

struct gguf_internal {
    struct gguf_kv* kvs;
};

static bool read_string(FILE* file, gguf_string_t* str) {
    if (fread(&str->len, sizeof(uint64_t), 1, file) != 1) return false;
    str->data = (char*)malloc(str->len + 1);
    if (!str->data) return false;
    if (fread(str->data, 1, str->len, file) != str->len) {
        free(str->data);
        return false;
    }
    str->data[str->len] = '\0';
    return true;
}

static void* read_value(FILE* file, gguf_value_type_t type) {
    void* value = NULL;
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            value = malloc(1);
            if (fread(value, 1, 1, file) != 1) { free(value); return NULL; }
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            value = malloc(2);
            if (fread(value, 2, 1, file) != 1) { free(value); return NULL; }
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            value = malloc(4);
            if (fread(value, 4, 1, file) != 1) { free(value); return NULL; }
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            value = malloc(8);
            if (fread(value, 8, 1, file) != 1) { free(value); return NULL; }
            break;
        case GGUF_TYPE_STRING: {
            gguf_string_t* str = (gguf_string_t*)malloc(sizeof(gguf_string_t));
            if (!read_string(file, str)) { free(str); return NULL; }
            value = str;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            gguf_array_t* arr = (gguf_array_t*)malloc(sizeof(gguf_array_t));
            uint32_t arr_type;
            if (fread(&arr_type, sizeof(uint32_t), 1, file) != 1) { free(arr); return NULL; }
            arr->type = (gguf_value_type_t)arr_type;
            if (fread(&arr->len, sizeof(uint64_t), 1, file) != 1) { free(arr); return NULL; }
            
            // Allocate pointers for array elements if it's string or array (though array of array might be complex)
            // For simplicity, we allocate array of values
            if (arr->len > 0) {
                // If it's a fixed size type, we can read it all at once or one by one
                // Better to read one by one to handle strings easily
                void** elements = (void**)malloc(arr->len * sizeof(void*));
                for (uint64_t i = 0; i < arr->len; i++) {
                    elements[i] = read_value(file, arr->type);
                }
                arr->data = elements;
            } else {
                arr->data = NULL;
            }
            value = arr;
            break;
        }
        default:
            return NULL;
    }
    return value;
}

static void free_value(gguf_value_type_t type, void* value) {
    if (!value) return;
    switch (type) {
        case GGUF_TYPE_STRING: {
            gguf_string_t* str = (gguf_string_t*)value;
            free(str->data);
            free(str);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            gguf_array_t* arr = (gguf_array_t*)value;
            if (arr->data) {
                void** elements = (void**)arr->data;
                for (uint64_t i = 0; i < arr->len; i++) {
                    free_value(arr->type, elements[i]);
                }
                free(elements);
            }
            free(arr);
            break;
        }
        default:
            free(value); // primitives
            break;
    }
}

gguf_file_t* gguf_file_read(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;

    gguf_file_t* f = (gguf_file_t*)malloc(sizeof(gguf_file_t));
    if (!f) {
        fclose(file);
        return NULL;
    }

    if (fread(&f->magic, sizeof(uint32_t), 1, file) != 1 || f->magic != GGUF_MAGIC) {
        free(f);
        fclose(file);
        return NULL;
    }

    if (fread(&f->version, sizeof(uint32_t), 1, file) != 1) goto error;
    if (fread(&f->tensor_count, sizeof(uint64_t), 1, file) != 1) goto error;
    if (fread(&f->metadata_kv_count, sizeof(uint64_t), 1, file) != 1) goto error;

    struct gguf_internal* internal = (struct gguf_internal*)malloc(sizeof(struct gguf_internal));
    if (!internal) goto error;
    
    internal->kvs = NULL;
    if (f->metadata_kv_count > 0) {
        internal->kvs = (struct gguf_kv*)malloc(f->metadata_kv_count * sizeof(struct gguf_kv));
        if (!internal->kvs) {
            free(internal);
            goto error;
        }

        for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
            if (!read_string(file, &internal->kvs[i].key)) {
                // Handle partial cleanup
                f->metadata_kv_count = i; // Update so free cleans up what was read
                f->internal_data = internal;
                gguf_file_free(f);
                fclose(file);
                return NULL;
            }
            uint32_t val_type;
            if (fread(&val_type, sizeof(uint32_t), 1, file) != 1) {
                f->metadata_kv_count = i;
                f->internal_data = internal;
                gguf_file_free(f);
                fclose(file);
                return NULL;
            }
            internal->kvs[i].type = (gguf_value_type_t)val_type;
            internal->kvs[i].value = read_value(file, internal->kvs[i].type);
            if (!internal->kvs[i].value) {
                free(internal->kvs[i].key.data);
                f->metadata_kv_count = i;
                f->internal_data = internal;
                gguf_file_free(f);
                fclose(file);
                return NULL;
            }
        }
    }

    f->internal_data = internal;
    fclose(file);
    return f;

error:
    free(f);
    fclose(file);
    return NULL;
}

void gguf_file_free(gguf_file_t* f) {
    if (!f) return;
    if (f->internal_data) {
        struct gguf_internal* internal = (struct gguf_internal*)f->internal_data;
        if (internal->kvs) {
            for (uint64_t i = 0; i < f->metadata_kv_count; i++) {
                free(internal->kvs[i].key.data);
                free_value(internal->kvs[i].type, internal->kvs[i].value);
            }
            free(internal->kvs);
        }
        free(internal);
    }
    free(f);
}

bool gguf_file_metadata(gguf_file_t* f, uint32_t index, const char** key, gguf_value_type_t* type, void** value) {
    if (!f || !f->internal_data || index >= f->metadata_kv_count) return false;
    
    struct gguf_internal* internal = (struct gguf_internal*)f->internal_data;
    if (key) *key = internal->kvs[index].key.data;
    if (type) *type = internal->kvs[index].type;
    if (value) *value = internal->kvs[index].value;
    
    return true;
}

// ---- Session 3 GGUF load API ----

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_DEFAULT_ALIGNMENT 32

static size_t read_string_gguf(const char *data, size_t offset, char **out_str) {
    uint64_t len = *(uint64_t *)(data + offset);
    offset += sizeof(uint64_t);
    *out_str = malloc(len + 1);
    memcpy(*out_str, data + offset, len);
    (*out_str)[len] = '\0';
    return offset + len;
}

static size_t skip_value_gguf(const char *data, size_t offset, gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return offset + 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return offset + 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return offset + 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return offset + 8;
        case GGUF_TYPE_STRING: {
            uint64_t len = *(uint64_t *)(data + offset);
            return offset + sizeof(uint64_t) + len;
        }
        case GGUF_TYPE_ARRAY: {
            gguf_type item_type = *(uint32_t *)(data + offset);
            offset += sizeof(uint32_t);
            uint64_t item_count = *(uint64_t *)(data + offset);
            offset += sizeof(uint64_t);
            for (uint64_t i = 0; i < item_count; i++) {
                offset = skip_value_gguf(data, offset, item_type);
            }
            return offset;
        }
        default: return offset;
    }
}

gguf_context* gguf_load(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    void *addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    gguf_context *ctx = calloc(1, sizeof(gguf_context));
    ctx->fd = fd;
    ctx->mmap_addr = addr;
    ctx->mmap_size = st.st_size;
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    const char *data = (const char *)addr;
    size_t offset = 0;

    memcpy(&ctx->header, data + offset, sizeof(gguf_header));
    offset += sizeof(gguf_header);

    if (strncmp(ctx->header.magic, "GGUF", 4) != 0) {
        gguf_free(ctx);
        return NULL;
    }

    ctx->kvs = calloc(ctx->header.kv_count, sizeof(gguf_kv_info));
    for (uint64_t i = 0; i < ctx->header.kv_count; i++) {
        offset = read_string_gguf(data, offset, &ctx->kvs[i].name);
        ctx->kvs[i].type = *(uint32_t *)(data + offset);
        offset += sizeof(uint32_t);
        ctx->kvs[i].value = (void *)(data + offset);
        if (strcmp(ctx->kvs[i].name, "general.alignment") == 0 && ctx->kvs[i].type == GGUF_TYPE_UINT32) {
            ctx->alignment = *(uint32_t *)ctx->kvs[i].value;
        }
        offset = skip_value_gguf(data, offset, ctx->kvs[i].type);
    }

    ctx->tensors = calloc(ctx->header.tensor_count, sizeof(gguf_tensor_info));
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        offset = read_string_gguf(data, offset, &ctx->tensors[i].name);
        ctx->tensors[i].ndim = *(uint32_t *)(data + offset);
        offset += sizeof(uint32_t);
        for (uint32_t j = 0; j < ctx->tensors[i].ndim; j++) {
            ctx->tensors[i].shape[j] = *(uint64_t *)(data + offset);
            offset += sizeof(uint64_t);
        }
        ctx->tensors[i].type = *(uint32_t *)(data + offset);
        offset += sizeof(uint32_t);
        ctx->tensors[i].offset = *(uint64_t *)(data + offset);
        offset += sizeof(uint64_t);
    }

    ctx->data_offset = (offset + ctx->alignment - 1) & ~(ctx->alignment - 1);
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        ctx->tensors[i].data = (void *)(data + ctx->data_offset + ctx->tensors[i].offset);
    }
    return ctx;
}

void gguf_free(gguf_context *ctx) {
    if (!ctx) return;
    if (ctx->kvs) {
        for (uint64_t i = 0; i < ctx->header.kv_count; i++) {
            free(ctx->kvs[i].name);
        }
        free(ctx->kvs);
    }
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
            free(ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }
    if (ctx->mmap_addr) {
        munmap(ctx->mmap_addr, ctx->mmap_size);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    free(ctx);
}

gguf_tensor_info* gguf_get_tensor(gguf_context *ctx, const char *name) {
    if (!ctx || !name) return NULL;
    for (uint64_t i = 0; i < ctx->header.tensor_count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}
