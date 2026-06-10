#pragma once

#include "../../topology/topology.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

struct ParallelBox {
    float minX, minY, maxX, maxY;
    int layer;
};

static inline ParallelBox getParallelBox(const GdsPolygon& poly) {
    ParallelBox box;
    if (poly.vertices.empty()) return {0.0f, 0.0f, 0.0f, 0.0f, poly.layer};

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

static inline bool isTooCloseParallel(const ParallelBox& a, const ParallelBox& b, float minDistSq) {
    float dx = std::max({0.0f, a.minX - b.maxX, b.minX - a.maxX});
    float dy = std::max({0.0f, a.minY - b.maxY, b.minY - a.maxY});

    return (dx * dx + dy * dy) < minDistSq;
}

int runParallelDistanceCheck(const Layout& layout, float minDistance) {
    int polygonCount = static_cast<int>(layout.polygons.size());
    if (polygonCount == 0) return 0;

    std::vector<ParallelBox> boxes(polygonCount);

    #pragma omp parallel for
    for (int i = 0; i < polygonCount; i++) {
        boxes[i] = getParallelBox(layout.polygons[i]);
    }

    int violations = 0;
    const float minDistSq = minDistance * minDistance;

    #pragma omp parallel for reduction(+:violations) schedule(guided)
    for (int i = 0; i < polygonCount; i++) {
        const ParallelBox& a = boxes[i];

        for (int j = i + 1; j < polygonCount; j++) {
            const ParallelBox& b = boxes[j];

            if (a.layer != b.layer) continue;

            if (isTooCloseParallel(a, b, minDistSq)) {
                violations++;
            }
        }
    }

    return violations;
}
