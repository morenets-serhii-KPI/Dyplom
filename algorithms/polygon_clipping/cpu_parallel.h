#pragma once

#include "../../topology/topology.h"
#include <vector>
#include <algorithm>
#include <omp.h>

struct BasicParallelClipBox {
    float minX, minY, maxX, maxY;
    int layer;
};

static inline BasicParallelClipBox getBasicParallelClipBox(const GdsPolygon& poly) {
    if (poly.vertices.empty()) return {0, 0, 0, 0, poly.layer};
    float x1 = poly.vertices[0].x, x2 = x1;
    float y1 = poly.vertices[0].y, y2 = y1;
    for (const auto& v : poly.vertices) {
        if (v.x < x1) x1 = v.x; else if (v.x > x2) x2 = v.x;
        if (v.y < y1) y1 = v.y; else if (v.y > y2) y2 = v.y;
    }
    return {x1, y1, x2, y2, poly.layer};
}

static inline bool intersectsBasicParallel(const BasicParallelClipBox& a, const BasicParallelClipBox& b) {
    return (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY);
}

static inline GdsPolygon buildBasicParallelIntersection(const BasicParallelClipBox& a, const BasicParallelClipBox& b) {
    GdsPolygon res;
    res.layer = a.layer;
    float ix1 = std::max(a.minX, b.minX);
    float iy1 = std::max(a.minY, b.minY);
    float ix2 = std::min(a.maxX, b.maxX);
    float iy2 = std::min(a.maxY, b.maxY);
    res.vertices = {{ix1, iy1}, {ix2, iy1}, {ix2, iy2}, {ix1, iy2}};
    return res;
}

inline Layout runParallelPolygonClipping(const Layout& layout) {
    int n = static_cast<int>(layout.polygons.size());
    if (n == 0) return Layout();

    std::vector<BasicParallelClipBox> boxes(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        boxes[i] = getBasicParallelClipBox(layout.polygons[i]);
    }

    Layout result;

    #pragma omp parallel
    {
        std::vector<GdsPolygon> localPolys;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            const auto& a = boxes[i];
            for (int j = i + 1; j < n; j++) {
                const auto& b = boxes[j];

                if (a.layer == b.layer && intersectsBasicParallel(a, b)) {
                    localPolys.push_back(buildBasicParallelIntersection(a, b));
                }
            }
        }

        #pragma omp critical
        {
            result.polygons.insert(
                result.polygons.end(),
                std::make_move_iterator(localPolys.begin()),
                std::make_move_iterator(localPolys.end())
            );
        }
    }

    return result;
}
