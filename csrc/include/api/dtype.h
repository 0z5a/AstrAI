#pragma once

#include <c10/core/ScalarType.h>

#include <datatype/element.cuh>

namespace astrai {

template <typename T> struct ScalarTypeOf;
template <> struct ScalarTypeOf<bf16> {
    static constexpr at::ScalarType value = at::kBFloat16;
};
template <> struct ScalarTypeOf<fp16> {
    static constexpr at::ScalarType value = at::kHalf;
};
template <> struct ScalarTypeOf<float> {
    static constexpr at::ScalarType value = at::kFloat;
};
template <> struct ScalarTypeOf<int8_t> {
    static constexpr at::ScalarType value = at::kChar;
};
template <> struct ScalarTypeOf<fp8_e4m3> {
    static constexpr at::ScalarType value = at::kFloat8_e4m3fn;
};
template <> struct ScalarTypeOf<fp8_e5m2> {
    static constexpr at::ScalarType value = at::kFloat8_e5m2;
};

template <typename T> inline constexpr at::ScalarType scalar_type_v = ScalarTypeOf<T>::value;

} // namespace astrai
