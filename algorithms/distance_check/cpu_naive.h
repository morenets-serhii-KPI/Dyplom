#pragma once

#include "../../topology/topology.h"
#include <cmath>
#include <algorithm>
#include <vector>

// Допоміжна структура для зберігання попередньо обчислених меж
struct BBox {
    float minX, minY, maxX, maxY;
    int layer;
};

static BBox computeBBox(const GdsPolygon& poly) {
    if (poly.vertices.empty()) return {0, 0, 0, 0, poly.layer};
    
    BBox b;
    b.minX = b.maxX = poly.vertices[0].x;
    b.minY = b.maxY = poly.vertices[0].y;
    b.layer = poly.layer;

    for (const auto& p : poly.vertices) {
        if (p.x < b.minX) b.minX = p.x;
        else if (p.x > b.maxX) b.maxX = p.x;
        if (p.y < b.minY) b.minY = p.y;
        else if (p.y > b.maxY) b.maxY = p.y;
    }
    return b;
}

int runNaiveDistanceCheck(const Layout& layout, float minDistance) {
    int violations = 0;
    size_t n = layout.polygons.size();
    if (n == 0) return 0;

    // 1. Попередньо обчислюємо всі межі (O(n))
    // Це прибирає дублювання роботи у вкладеному циклі
    std::vector<BBox> boxes;
    boxes.reserve(n);
    for (const auto& poly : layout.polygons) {
        boxes.push_back(computeBBox(poly));
    }

    float minDistSq = minDistance * minDistance;

    // 2. Основний цикл порівняння (O(n^2))
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const BBox& a = boxes[i];
            const BBox& b = boxes[j];

            // Фільтр по шарах
            if (a.layer != b.layer) continue;

            // Швидка перевірка на перетин або близькість по осях
            float dx = std::max({0.0f, a.minX - b.maxX, b.minX - a.maxX});
            float dy = std::max({0.0f, a.minY - b.maxY, b.minY - a.maxY});

            // Порівнюємо квадрати відстаней (уникаємо sqrt)
            if (dx * dx + dy * dy < minDistSq) {
                violations++;
            }
        }
    }

    return violations;
}