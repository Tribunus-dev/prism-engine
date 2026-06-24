add_executable(test_kv_cache test_kv_cache.c)
target_link_libraries(test_kv_cache PRIVATE mlxc)
add_test(NAME test_kv_cache COMMAND test_kv_cache)
