#pragma once

#include "../../topology/topology.h"
#include <algorithm>
#include <vector>

struct ClipBox {
    float minX, minY, maxX, maxY;
    int layer;
};

static inline ClipBox getClipBox(const GdsPolygon& poly) {
    ClipBox box;
    if (poly.vertices.empty()) return {0,0,0,0, poly.layer};
    
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

static inline bool intersects(const ClipBox& a, const ClipBox& b) {
    return (a.minX < b.maxX && a.maxX > b.minX &&
            a.minY < b.maxY && a.maxY > b.minY);
}

Layout runNaivePolygonClipping(const Layout& layout) {
    Layout result;
    size_t n = layout.polygons.size();
    if (n == 0) return result;

    // 1. Precompute boxes (O(n))
    std::vector<ClipBox> boxes(n);
    for (size_t i = 0; i < n; ++i) {
        boxes[i] = getClipBox(layout.polygons[i]);
    }

    // 2. Наївний пошук перетинів (O(n^2))
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (boxes[i].layer != boxes[j].layer) continue;

            if (intersects(boxes[i], boxes[j])) {
                const auto& a = boxes[i];
                const auto& b = boxes[j];
                
                GdsPolygon clipped;
                clipped.layer = a.layer;
                
                float ix1 = std::max(a.minX, b.minX);
                float iy1 = std::max(a.minY, b.minY);
                float ix2 = std::min(a.maxX, b.maxX);
                float iy2 = std::min(a.maxY, b.maxY);

                clipped.vertices = {
                    {ix1, iy1}, {ix2, iy1}, {ix2, iy2}, {ix1, iy2}
                };
                
                result.polygons.push_back(std::move(clipped));
            }
        }
    }

    return result;
}