#pragma once

#include <vector>
#include <fstream>
#include <random>

#include "../external/json.hpp"

using json = nlohmann::json;

struct Point {
    float x;
    float y;
};

struct GdsPolygon {

    int layer;
    std::vector<Point> vertices;

    static GdsPolygon createRectangle(
        float x,
        float y,
        float width,
        float height,
        int layer
    ) {

        GdsPolygon rect;

        rect.layer = layer;

        rect.vertices.push_back({x, y});
        rect.vertices.push_back({x + width, y});
        rect.vertices.push_back({x + width, y + height});
        rect.vertices.push_back({x, y + height});

        return rect;
    }
};

struct Layout {

    std::vector<GdsPolygon> polygons;
};

struct Scene {

    std::vector<Layout> layouts;

    void generateRandomLayout(
        int polygonCount,
        int layerCount,
        float worldWidth,
        float worldHeight,
        float minSize,
        float maxSize,
        unsigned int seed = 42
    ) {

        layouts.clear();

        Layout layout;

        std::mt19937 rng(seed);

        std::uniform_real_distribution<float> posX(
            0.0f,
            worldWidth
        );

        std::uniform_real_distribution<float> posY(
            0.0f,
            worldHeight
        );

        std::uniform_real_distribution<float> sizeDist(
            minSize,
            maxSize
        );

        std::uniform_int_distribution<int> layerDist(
            1,
            layerCount
        );

        for (int i = 0; i < polygonCount; i++) {

            float x = posX(rng);
            float y = posY(rng);

            float width = sizeDist(rng);
            float height = sizeDist(rng);

            int layer = layerDist(rng);

            GdsPolygon rect = GdsPolygon::createRectangle(
                x,
                y,
                width,
                height,
                layer
            );

            layout.polygons.push_back(rect);
        }

        layouts.push_back(layout);
    }

    void exportToJson(const char* filename) const {

        std::ofstream file(filename);

        file << "[\n";

        bool firstPolygon = true;

        for (const auto& layout : layouts) {

            for (const auto& polygon : layout.polygons) {

                if (!firstPolygon)
                    file << ",\n";

                firstPolygon = false;

                file << "  {\n";

                file << "    \"layer\": "
                     << polygon.layer
                     << ",\n";

                file << "    \"datatype\": 0,\n";

                file << "    \"points\": [\n";

                for (size_t i = 0;
                     i < polygon.vertices.size();
                     i++) {

                    const Point& p =
                        polygon.vertices[i];

                    file << "      ["
                         << p.x
                         << ", "
                         << p.y
                         << "]";

                    if (i != polygon.vertices.size() - 1)
                        file << ",";

                    file << "\n";
                }

                file << "    ]\n";

                file << "  }";
            }
        }

        file << "\n]\n";

        file.close();
    }

    void importFromJson(const char* filename) {

        layouts.clear();

        Layout layout;

        std::ifstream file(filename);

        nlohmann::json data;

        file >> data;

        for (const auto& polyJson : data) {

            GdsPolygon polygon;

            polygon.layer = polyJson["layer"];

            for (const auto& pointJson :
                polyJson["points"]) {

                Point point;

                point.x = pointJson[0];
                point.y = pointJson[1];

                polygon.vertices.push_back(point);
            }

            layout.polygons.push_back(polygon);
        }

        layouts.push_back(layout);
    }
};