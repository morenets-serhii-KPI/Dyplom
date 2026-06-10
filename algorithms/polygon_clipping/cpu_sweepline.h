#pragma once

#include "../../topology/topology.h"
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>

struct SweepClippingBox {
    float minX, minY, maxX, maxY;
    int layer;
};

struct ClippingEvent {
    float x;
    bool isStart;
    size_t boxIdx;

    bool operator<(const ClippingEvent& other) const {
        if (std::abs(x - other.x) > 1e-7f) return x < other.x;
        return isStart > other.isStart;
    }
};

static inline SweepClippingBox getSweepClippingBox(const GdsPolygon& poly) {
    SweepClippingBox box;
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

static inline bool boxesOverlap(const SweepClippingBox& a, const SweepClippingBox& b) {
    return (a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY);
}

struct YClipComp {
    const std::vector<SweepClippingBox>& boxes;
    YClipComp(const std::vector<SweepClippingBox>& b) : boxes(b) {}
    bool operator()(size_t i, size_t j) const {
        if (std::abs(boxes[i].minY - boxes[j].minY) > 1e-7f)
            return boxes[i].minY < boxes[j].minY;
        return i < j;
    }
};

inline Layout runSweepLinePolygonClipping(const Layout& layout) {
    if (layout.polygons.empty()) return Layout();

    size_t n = layout.polygons.size();
    std::vector<SweepClippingBox> boxes(n);
    std::vector<ClippingEvent> events;
    events.reserve(n * 2);

    for (size_t i = 0; i < n; i++) {
        boxes[i] = getSweepClippingBox(layout.polygons[i]);
        events.push_back({boxes[i].minX, true, i});
        events.push_back({boxes[i].maxX, false, i});
    }

    std::sort(events.begin(), events.end());

    YClipComp comp(boxes);
    std::set<size_t, YClipComp> activeSet(comp);
    Layout result;

    for (const auto& ev : events) {
        if (ev.isStart) {
            const auto& current = boxes[ev.boxIdx];
            for (size_t otherIdx : activeSet) {
                const auto& other = boxes[otherIdx];
                if (current.layer == other.layer && boxesOverlap(current, other)) {
                    float ix1 = std::max(current.minX, other.minX);
                    float iy1 = std::max(current.minY, other.minY);
                    float ix2 = std::min(current.maxX, other.maxX);
                    float iy2 = std::min(current.maxY, other.maxY);

                    GdsPolygon clip;
                    clip.layer = current.layer;
                    clip.vertices = {{ix1, iy1}, {ix2, iy1}, {ix2, iy2}, {ix1, iy2}};
                    result.polygons.push_back(std::move(clip));
                }
            }
            activeSet.insert(ev.boxIdx);
        } else {
            activeSet.erase(ev.boxIdx);
        }
    }
    return result;
}
