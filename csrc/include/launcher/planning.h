#pragma once
/* Host planner interface; implementation is compiled once in gemm/planning.cpp. */
#include <vector>

#include <launcher/plan_types.h>

namespace astrai {
namespace gemm {

int plan_raster(const PlanQuery& q, int bm, int bn);
std::vector<GemmRecipe> gemm_recipes_for(bool crosswise_staging, int ba, int bb);

} // namespace gemm
} // namespace astrai
