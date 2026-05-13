#pragma once

#include "../../topology/topology.h"
#include <vector>
#include <algorithm>
#include <cmath>

struct SweepBox {
    float minX, minY, maxX, maxY;
    int layer;
};

struct SweepEvent {
    float x;
    bool isStart;
    size_t boxIdx;

    bool operator<(const SweepEvent& other) const {
        if (std::abs(x - other.x) > 1e-7f) return x < other.x;
        return isStart > other.isStart; // Start перед End
    }
};

static inline SweepBox getBox(const GdsPolygon& poly) {
    SweepBox box;
    if (poly.vertices.empty()) return {0,0,0,0, poly.layer};
    box.minX = box.maxX = poly.vertices[0].x;
    box.minY = box.maxY = poly.vertices[0].y;
    box.layer = poly.layer;
    for (const auto& p : poly.vertices) {
        if (p.x < box.minX) box.minX = p.x; else if (p.x > box.maxX) box.maxX = p.x;
        if (p.y < box.minY) box.minY = p.y; else if (p.y > box.maxY) box.maxY = p.y;
    }
    return box;
}

int runSweepLineDistanceCheck(const Layout& layout, float minDistance) {
    size_t n = layout.polygons.size();
    if (n == 0) return 0;

    std::vector<SweepBox> boxes(n);
    std::vector<SweepEvent> events;
    events.reserve(n * 2);

    for (size_t i = 0; i < n; i++) {
        boxes[i] = getBox(layout.polygons[i]);
        events.push_back({boxes[i].minX, true, i});
        events.push_back({boxes[i].maxX + minDistance, false, i});
    }

    std::sort(events.begin(), events.end());

    // Замість std::set використовуємо vector для activeSet (менше алокацій)
    // Для дипломної роботи: на невеликих "активних наборах" vector швидший за дерево
    std::vector<size_t> activeIndices;
    int violations = 0;
    const float minDistSq = minDistance * minDistance;

    for (const auto& ev : events) {
        size_t curIdx = ev.boxIdx;
        const auto& curBox = boxes[curIdx];

        if (ev.isStart) {
            for (size_t otherIdx : activeIndices) {
                const auto& otherBox = boxes[otherIdx];
                
                if (curBox.layer != otherBox.layer) continue;

                // Оскільки ми вже відфільтровані по X завдяки sweep-line,
                // додаємо швидку перевірку по Y перед розрахунком відстані
                float dy = std::max({0.0f, curBox.minY - otherBox.maxY, otherBox.minY - curBox.maxY});
                if (dy >= minDistance) continue;

                float dx = std::max({0.0f, curBox.minX - otherBox.maxX, otherBox.minX - curBox.maxX});
                if (dx * dx + dy * dy < minDistSq) {
                    violations++;
                }
            }
            activeIndices.push_back(curIdx);
        } else {
            // Видалення з вектора (O(k), де k - розмір активного набору)
            auto it = std::find(activeIndices.begin(), activeIndices.end(), curIdx);
            if (it != activeIndices.end()) {
                *it = activeIndices.back();
                activeIndices.pop_back();
            }
        }
    }
    return violations;
}