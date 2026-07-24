#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "arena_info.h"

static NSArray<NSNumber *> *shapeFor(const ArenaInfo *a) {
    if (a->logical_dim0 > 0 && a->logical_dim1 > 0)
        return @[@(a->logical_dim0), @(a->logical_dim1)];
    if (a->logical_dim0 > 0) return @[@(a->logical_dim0)];
    return @[@(a->byte_size / (int)sizeof(float))];
}

extern "C" int tribunus_coreml_load_model(void **out_model, const char *path, long long units) {
    @autoreleasepool {
        if (!out_model || !path) return -1;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = (MLComputeUnits)units;
        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
        if (!model) return error ? (int)error.code : -2;
        *out_model = (__bridge_retained void *)model;
        return 0;
    }
}

extern "C" void tribunus_coreml_free_model(void *ptr) {
    if (ptr) CFRelease((CFTypeRef)ptr);
}

extern "C" int tribunus_coreml_predict(void *ptr, const char *input_name,
    const ArenaInfo *input, const char *output_name, ArenaInfo *output) {
    @autoreleasepool {
        if (!ptr || !input || !output || !input_name || !output_name) return -1;
        MLModel *model = (__bridge MLModel *)ptr;
        NSError *error = nil;
        MLMultiArray *inputArray = [[MLMultiArray alloc] initWithShape:shapeFor(input)
            dataType:MLMultiArrayDataTypeFloat32 error:&error];
        if (!inputArray || error) return error ? (int)error.code : -2;
        memcpy(inputArray.dataPointer, input->base_address, (size_t)input->byte_size);
        MLFeatureValue *inputValue = [MLFeatureValue featureValueWithMultiArray:inputArray];
        NSDictionary *features = @{[NSString stringWithUTF8String:input_name]: inputValue};
        id<MLFeatureProvider> result = [model predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error] error:&error];
        if (!result) return error ? (int)error.code : -2;
        MLFeatureValue *outputValue = [result featureValueForName:[NSString stringWithUTF8String:output_name]];
        MLMultiArray *outputArray = outputValue.multiArrayValue;
        if (!outputArray || !output->base_address) return -3;
        NSUInteger bytes = outputArray.count * sizeof(float);
        if (bytes > (NSUInteger)output->byte_size) return -4;
        memcpy(output->base_address, outputArray.dataPointer, bytes);
        output->byte_size = (int)bytes;
        return 0;
    }
}

extern "C" int tribunus_coreml_predict_two(void *ptr, const char *name_a, const ArenaInfo *a,
    const char *name_b, const ArenaInfo *b, const char *output_name, ArenaInfo *output) {
    @autoreleasepool {
        if (!ptr || !a || !b || !output || !name_a || !name_b || !output_name) return -1;
        MLModel *model = (__bridge MLModel *)ptr;
        NSError *error = nil;
        MLMultiArray *array_a = [[MLMultiArray alloc] initWithShape:shapeFor(a) dataType:MLMultiArrayDataTypeFloat32 error:&error];
        if (!array_a || error) return error ? (int)error.code : -2;
        MLMultiArray *array_b = [[MLMultiArray alloc] initWithShape:shapeFor(b) dataType:MLMultiArrayDataTypeFloat32 error:&error];
        if (!array_b || error) return error ? (int)error.code : -3;
        memcpy(array_a.dataPointer, a->base_address, (size_t)a->byte_size);
        memcpy(array_b.dataPointer, b->base_address, (size_t)b->byte_size);
        NSDictionary *features = @{
            [NSString stringWithUTF8String:name_a]: [MLFeatureValue featureValueWithMultiArray:array_a],
            [NSString stringWithUTF8String:name_b]: [MLFeatureValue featureValueWithMultiArray:array_b]
        };
        id<MLFeatureProvider> result = [model predictionFromFeatures:[[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error] error:&error];
        if (!result) return error ? (int)error.code : -4;
        MLMultiArray *array_out = [[result featureValueForName:[NSString stringWithUTF8String:output_name]] multiArrayValue];
        if (!array_out || !output->base_address) return -5;
        NSUInteger bytes = array_out.count * sizeof(float);
        if (bytes > (NSUInteger)output->byte_size) return -6;
        memcpy(output->base_address, array_out.dataPointer, bytes);
        output->byte_size = (int)bytes;
        return 0;
    }
}
