#pragma once

#include "../../topology/topology.h"
#include <vector>
#include <algorithm>
#include <omp.h>

// Використовуємо унікальне ім'я, щоб не конфліктувати з іншими блоками
struct ParallelClippingBox {
    float minX, minY, maxX, maxY;
    int layer;
};

static inline ParallelClippingBox getParallelClipBox(const GdsPolygon& poly) {
    if (poly.vertices.empty()) return {0, 0, 0, 0, poly.layer};
    
    float x1 = poly.vertices[0].x, x2 = x1;
    float y1 = poly.vertices[0].y, y2 = y1;
    
    for (const auto& v : poly.vertices) {
        if (v.x < x1) x1 = v.x; else if (v.x > x2) x2 = v.x;
        if (v.y < y1) y1 = v.y; else if (v.y > y2) y2 = v.y;
    }
    return {x1, y1, x2, y2, poly.layer};
}

static inline bool intersectsParallel(const ParallelClippingBox& a, const ParallelClippingBox& b) {
    return (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY);
}

// Функція генерації полігону перетину
static inline GdsPolygon createIntersectionPoly(const ParallelClippingBox& a, const ParallelClippingBox& b) {
    GdsPolygon res;
    res.layer = a.layer;
    float ix1 = std::max(a.minX, b.minX);
    float iy1 = std::max(a.minY, b.minY);
    float ix2 = std::min(a.maxX, b.maxX);
    float iy2 = std::min(a.maxY, b.maxY);
    res.vertices = {{ix1, iy1}, {ix2, iy1}, {ix2, iy2}, {ix1, iy2}};
    return res;
}

inline Layout runOptimizedParallelPolygonClipping(const Layout& layout) {
    int n = static_cast<int>(layout.polygons.size());
    if (n == 0) return Layout();

    std::vector<ParallelClippingBox> boxes(n);

    // 1. Паралельне обчислення Bounding Boxes
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        boxes[i] = getParallelClipBox(layout.polygons[i]);
    }

    // 2. Сортування по X для Sweep-line ефекту
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) {
        return a.minX < b.minX;
    });

    Layout result;
    
    // 3. Паралельний пошук перетинів
    #pragma omp parallel
    {
        std::vector<GdsPolygon> localPolys;
        // Резервуємо трохи місця, щоб зменшити кількість реаллокацій
        localPolys.reserve(n / omp_get_num_threads());

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            const auto& a = boxes[i];
            for (int j = i + 1; j < n; j++) {
                const auto& b = boxes[j];

                // Sweep-line break: якщо ліва межа B вже правіше правої межі A + 0 (для кліпінгу)
                if (b.minX > a.maxX) break;

                if (a.layer == b.layer && intersectsParallel(a, b)) {
                    localPolys.push_back(createIntersectionPoly(a, b));
                }
            }
        }

        // 4. Злиття результатів (Critical section)
        #pragma omp critical
        {
            // Використовуємо move-ітератори для ефективності
            result.polygons.insert(
                result.polygons.end(),
                std::make_move_iterator(localPolys.begin()),
                std::make_move_iterator(localPolys.end())
            );
        }
    }

    return result;
}