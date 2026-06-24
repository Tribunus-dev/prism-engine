#include "../include/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void write_test_gguf(const char* filename) {
    FILE* f = fopen(filename, "wb");
    assert(f != NULL);

    uint32_t magic = GGUF_MAGIC;
    fwrite(&magic, sizeof(uint32_t), 1, f);

    uint32_t version = 3;
    fwrite(&version, sizeof(uint32_t), 1, f);

    uint64_t tensor_count = 0;
    fwrite(&tensor_count, sizeof(uint64_t), 1, f);

    uint64_t kv_count = 2; // Two metadata KV pairs
    fwrite(&kv_count, sizeof(uint64_t), 1, f);

    // KV 1: "general.architecture" -> String "llama"
    const char* key1 = "general.architecture";
    uint64_t key1_len = strlen(key1);
    fwrite(&key1_len, sizeof(uint64_t), 1, f);
    fwrite(key1, 1, key1_len, f);

    uint32_t type1 = GGUF_TYPE_STRING;
    fwrite(&type1, sizeof(uint32_t), 1, f);

    const char* val1 = "llama";
    uint64_t val1_len = strlen(val1);
    fwrite(&val1_len, sizeof(uint64_t), 1, f);
    fwrite(val1, 1, val1_len, f);

    // KV 2: "llama.context_length" -> UINT32 4096
    const char* key2 = "llama.context_length";
    uint64_t key2_len = strlen(key2);
    fwrite(&key2_len, sizeof(uint64_t), 1, f);
    fwrite(key2, 1, key2_len, f);

    uint32_t type2 = GGUF_TYPE_UINT32;
    fwrite(&type2, sizeof(uint32_t), 1, f);

    uint32_t val2 = 4096;
    fwrite(&val2, sizeof(uint32_t), 1, f);

    fclose(f);
}

int main() {
    const char* filename = "test_temp.gguf";
    write_test_gguf(filename);

    gguf_file_t* f = gguf_file_read(filename);
    assert(f != NULL);
    assert(f->magic == GGUF_MAGIC);
    assert(f->version == 3);
    assert(f->tensor_count == 0);
    assert(f->metadata_kv_count == 2);

    const char* key;
    gguf_value_type_t type;
    void* value;

    // Read KV 1
    assert(gguf_file_metadata(f, 0, &key, &type, &value));
    assert(strcmp(key, "general.architecture") == 0);
    assert(type == GGUF_TYPE_STRING);
    gguf_string_t* str_val = (gguf_string_t*)value;
    assert(str_val->len == 5);
    assert(strcmp(str_val->data, "llama") == 0);

    // Read KV 2
    assert(gguf_file_metadata(f, 1, &key, &type, &value));
    assert(strcmp(key, "llama.context_length") == 0);
    assert(type == GGUF_TYPE_UINT32);
    uint32_t u32_val = *(uint32_t*)value;
    assert(u32_val == 4096);

    gguf_file_free(f);
    remove(filename);

    printf("All tests passed!\n");
    return 0;
}
