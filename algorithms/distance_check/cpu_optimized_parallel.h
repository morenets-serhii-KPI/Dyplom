#pragma once

#include "../../topology/topology.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

struct OptimizedParallelBox {
    float minX, minY, maxX, maxY;
    int layer;
};

static OptimizedParallelBox getOptimizedParallelBox(const GdsPolygon& poly) {
    OptimizedParallelBox box;
    if (poly.vertices.empty()) return {0, 0, 0, 0, poly.layer};

    box.minX = box.maxX = poly.vertices[0].x;
    box.minY = box.maxY = poly.vertices[0].y;
    box.layer = poly.layer;

    for (const auto& p : poly.vertices) {
        if (p.x < box.minX) box.minX = p.x;
        else if (p.x > box.maxX) box.maxX = p.x;
        if (p.y < box.minY) box.minY = p.y;
        else if (p.y > box.maxY) box.maxY = p.y;
    }
    return box;
}

static inline bool isTooCloseOptimizedParallel(const OptimizedParallelBox& a, const OptimizedParallelBox& b, float minDistSq) {
    float dx = std::max({0.0f, a.minX - b.maxX, b.minX - a.maxX});
    float dy = std::max({0.0f, a.minY - b.maxY, b.minY - a.maxY});

    return (dx * dx + dy * dy) < minDistSq;
}

int runOptimizedParallelDistanceCheck(const Layout& layout, float minDistance) {
    int polygonCount = static_cast<int>(layout.polygons.size());
    if (polygonCount == 0) return 0;

    std::vector<OptimizedParallelBox> boxes(polygonCount);

    #pragma omp parallel for
    for (int i = 0; i < polygonCount; i++) {
        boxes[i] = getOptimizedParallelBox(layout.polygons[i]);
    }

    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) {
        return a.minX < b.minX;
    });

    int violations = 0;
    const float minDistSq = minDistance * minDistance;

    #pragma omp parallel for reduction(+:violations) schedule(dynamic)
    for (int i = 0; i < polygonCount; i++) {
        const auto& a = boxes[i];

        for (int j = i + 1; j < polygonCount; j++) {
            const auto& b = boxes[j];

            if (b.minX - a.maxX > minDistance) {
                break;
            }

            if (a.layer == b.layer) {
                if (isTooCloseOptimizedParallel(a, b, minDistSq)) {
                    violations++;
                }
            }
        }
    }

    return violations;
}
