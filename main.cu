#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "topology/topology.h"

/*
    DISTANCE CHECK
*/

#include "algorithms/distance_check/cpu_naive.h"
#include "algorithms/distance_check/cpu_sweepline.h"
#include "algorithms/distance_check/cpu_parallel.h"
#include "algorithms/distance_check/cpu_optimized_parallel.h"

#include "algorithms/distance_check/gpu_parallel.cuh"
#include "algorithms/distance_check/gpu_optimized.cuh"

/*
    POLYGON CLIPPING
*/

#include "algorithms/polygon_clipping/cpu_naive.h"
#include "algorithms/polygon_clipping/cpu_sweepline.h"
#include "algorithms/polygon_clipping/cpu_parallel.h"
#include "algorithms/polygon_clipping/cpu_optimized_parallel.h"

#include "algorithms/polygon_clipping/gpu_parallel.cuh"
#include "algorithms/polygon_clipping/gpu_optimized.cuh"

template<typename Func, typename... Args>
auto measureTime(
    Func func,
    Args&&... args
) {

    auto start =
        std::chrono
        ::high_resolution_clock
        ::now();

    auto result =
        func(
            std::forward<Args>(args)...
        );

    auto end =
        std::chrono
        ::high_resolution_clock
        ::now();

    long long ms =

        std::chrono
        ::duration_cast
        <
            std::chrono::milliseconds
        >
        (
            end - start
        )
        .count();

    return std::make_pair(
        result,
        ms
    );
}

void printSeparator() {

    std::cout
        << "\n"
        << std::string(100, '=')
        << "\n";
}

struct DensityConfig {

    std::string name;

    int width;
    int height;
};

int main() {

    std::vector<int> polygonCounts = {

        1000,
        5000,
        10000
    };

    /*
        3 DENSITY LEVELS
    */

    std::vector<DensityConfig> densities = {

        {
            "LOW DENSITY",
            30000,
            30000
        },

        {
            "MEDIUM DENSITY",
            10000,
            10000
        },

        {
            "HIGH DENSITY",
            3000,
            3000
        }
    };

    float minDistance =
        20.0f;

    /*
        TEST LOOP
    */

    for (
        const auto& density
        :
        densities
    ) {

        printSeparator();

        std::cout
            << density.name
            << "\n";

        std::cout
            << "Layout Size: "
            << density.width
            << " x "
            << density.height
            << "\n";

        printSeparator();

        /*
            GENERATE LAYOUTS
        */

        std::vector<Layout>
            layouts;

        for (
            int count
            :
            polygonCounts
        ) {

            Scene scene;

            scene.generateRandomLayout(

                count,

                3,

                density.width,
                density.height,

                20,
                120
            );

            layouts.push_back(
                scene.layouts[0]
            );
        }

        /*
            DISTANCE CHECK
        */

        printSeparator();

        std::cout
            << "DISTANCE CHECK\n";

        printSeparator();

        for (

            size_t i = 0;

            i < polygonCounts.size();

            i++
        ) {

            int count =
                polygonCounts[i];

            Layout& layout =
                layouts[i];

            auto cpuNaive =

                measureTime(

                    runNaiveDistanceCheck,

                    layout,

                    minDistance
                );

            auto cpuSweep =

                measureTime(

                    runSweepLineDistanceCheck,

                    layout,

                    minDistance
                );

            auto cpuParallel =

                measureTime(

                    runParallelDistanceCheck,

                    layout,

                    minDistance
                );

            auto cpuOptimized =

                measureTime(

                    runOptimizedParallelDistanceCheck,

                    layout,

                    minDistance
                );

            auto gpuParallel =

                measureTime(

                    runGpuBruteforceDistanceCheck,

                    layout,

                    minDistance
                );

            auto gpuOptimized =

                measureTime(

                    runGpuOptimizedDistanceCheck,

                    layout,

                    minDistance
                );

            std::cout
                << "\nPolygons: "
                << count
                << "\n\n";

            std::cout
                << "CPU Naive:                 "
                << cpuNaive.second
                << " ms | "
                << cpuNaive.first
                << "\n";

            std::cout
                << "CPU SweepLine:             "
                << cpuSweep.second
                << " ms | "
                << cpuSweep.first
                << "\n";

            std::cout
                << "CPU Parallel:              "
                << cpuParallel.second
                << " ms | "
                << cpuParallel.first
                << "\n";

            std::cout
                << "CPU Optimized Parallel:    "
                << cpuOptimized.second
                << " ms | "
                << cpuOptimized.first
                << "\n";

            std::cout
                << "GPU Parallel:              "
                << gpuParallel.second
                << " ms | "
                << gpuParallel.first
                << "\n";

            std::cout
                << "GPU Optimized:             "
                << gpuOptimized.second
                << " ms | "
                << gpuOptimized.first
                << "\n";
        }

        /*
            POLYGON CLIPPING
        */

        printSeparator();

        std::cout
            << "POLYGON CLIPPING\n";

        printSeparator();

        for (

            size_t i = 0;

            i < polygonCounts.size();

            i++
        ) {

            int count =
                polygonCounts[i];

            Layout& layout =
                layouts[i];

            auto cpuNaive =

                measureTime(

                    runNaivePolygonClipping,

                    layout
                );

            auto cpuSweep =

                measureTime(

                    runSweepLinePolygonClipping,

                    layout
                );

            auto cpuParallel =

                measureTime(

                    runParallelPolygonClipping,

                    layout
                );

            auto cpuOptimized =

                measureTime(

                    runOptimizedParallelPolygonClipping,

                    layout
                );

            auto gpuParallel =

                measureTime(

                    runGpuPolygonClipping,

                    layout
                );

            auto gpuOptimized =

                measureTime(

                    runGpuOptimizedPolygonClipping,

                    layout
                );

            std::cout
                << "\nPolygons: "
                << count
                << "\n\n";

            std::cout
                << "CPU Naive:                 "
                << cpuNaive.second
                << " ms | "
                << cpuNaive.first.polygons.size()
                << "\n";

            std::cout
                << "CPU SweepLine:             "
                << cpuSweep.second
                << " ms | "
                << cpuSweep.first.polygons.size()
                << "\n";

            std::cout
                << "CPU Parallel:              "
                << cpuParallel.second
                << " ms | "
                << cpuParallel.first.polygons.size()
                << "\n";

            std::cout
                << "CPU Optimized Parallel:    "
                << cpuOptimized.second
                << " ms | "
                << cpuOptimized.first.polygons.size()
                << "\n";

            std::cout
                << "GPU Parallel:              "
                << gpuParallel.second
                << " ms | "
                << gpuParallel.first.polygons.size()
                << "\n";

            std::cout
                << "GPU Optimized:             "
                << gpuOptimized.second
                << " ms | "
                << gpuOptimized.first.polygons.size()
                << "\n";
        }
    }

    printSeparator();

    std::cout
        << "ALL TESTS COMPLETED\n";

    printSeparator();

    return 0;
}