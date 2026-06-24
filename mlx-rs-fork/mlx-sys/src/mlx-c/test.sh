#!/bin/bash
gcc -Iinclude tests/test_llama.c src/llama.c -lm -o test_llama && ./test_llama
