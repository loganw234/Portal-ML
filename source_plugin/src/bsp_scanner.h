#pragma once

#include <string>
#include <vector>

#include "blackboard_extractor.h"

namespace portal_ml {

struct DiscoveredEntity {
    std::string id;
    std::string kind;  // "cube" | "floor_button" | "pedestal_button" | "exit_door" | "fizzler" | "elevator"
    std::string classname;
    std::string model;
    Vec3 spawn_pos{};
};

struct HazardRecord {
    std::string id;
    std::string kind;  // "acid" | "trigger_hurt"
    Vec3 min_pos{};
    Vec3 max_pos{};
    float dps = 100.0f;
};

struct BspMapData {
    std::string map_name;
    bool loaded = false;
    int bsp_version = 0;
    bool can_fire_blue = true;
    bool can_fire_orange = true;
    Vec3 player_spawn{};
    float player_spawn_yaw = 0.0f;
    std::vector<DiscoveredEntity> entities;
    std::vector<PortalWallRecord> portal_surfaces;
    std::vector<HazardRecord> hazards;
};

// Pure C++17 Source Engine BSP (v19/v20) scanner for generic live-map discovery.
// Parses Lump 0 (entities) and geometry/texdata lumps (1,2,3,6,7,12,13,43,44)
// to extract gameplay entities and portalable concrete surfaces from arbitrary
// Valve Portal 1 maps without hardcoded chamber coordinates.
class BspScanner {
public:
    static bool LoadFromFile(const std::string& bsp_path, const std::string& map_name,
                             BspMapData& out, float min_extent = 56.0f);
};

}  // namespace portal_ml
