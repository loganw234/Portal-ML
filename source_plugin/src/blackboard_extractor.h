#pragma once

#include <string>
#include <vector>

namespace portal_ml {

struct BspMapData;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct ZoneRect {
    int zone_id = 0;
    float min_x = 0.0f;
    float max_x = 0.0f;
    float min_y = 0.0f;
    float max_y = 0.0f;
};

struct ChamberLayout {
    int chamber_id = 1;
    float min_y = 0.0f;
    float max_y = 256.0f;
    Vec3 player_spawn;
    float player_yaw = 0.0f;
    bool exit_starts_open = false;
    bool has_cube = false;
    std::string cube_id;
    Vec3 cube_spawn;
    bool has_floor_button = false;
    std::string floor_button_id;
    Vec3 floor_button_pos;
    bool has_pedestal = false;
    std::string pedestal_id;
    Vec3 pedestal_pos;
    std::string exit_id;
    Vec3 exit_pos;
    int max_steps = 200;
    std::vector<ZoneRect> zones;
};

struct EntityRecord {
    std::string id;
    std::string kind;  // "cube" | "floor_button" | "pedestal_button" | "exit_door"
    Vec3 pos;
    Vec3 vel;
    int zone_id = 0;
    bool active = false;
    bool held = false;
    float timer = 0.0f;
};

struct PortalWallRecord {
    std::string id;
    Vec3 pos;
    Vec3 normal;
    int zone_id = 0;
    float width = 128.0f;
    float height = 128.0f;
};

struct PortalRecord {
    std::string color;  // "blue" | "orange"
    bool active = false;
    std::string wall_id;
    Vec3 pos;
    Vec3 normal;
    int zone_id = -1;
};

struct BlackboardState {
    int tick = 0;
    float dt = 0.05f;
    int chamber_id = 1;
    Vec3 player_pos;
    Vec3 player_vel;
    float player_yaw = 0.0f;
    float player_pitch = 0.0f;
    int player_zone_id = 0;
    std::string held_entity_id;
    std::vector<EntityRecord> entities;
    std::vector<PortalWallRecord> portal_walls;
    PortalRecord blue_portal{"blue"};
    PortalRecord orange_portal{"orange"};
    bool done = false;
    bool success = false;
};

const ChamberLayout& GetChamberLayout(int chamber_id);
std::vector<PortalWallRecord> GetChamberPortalWalls(int chamber_id);
int ClassifyZone(const ChamberLayout& layout, float x, float y);

// Extracts ground-truth symbolic world state from Source Engine server entities
// and serializes it to the exact JSON schema consumed by portal_ml.schema.BlackboardSnapshot.
class BlackboardExtractor {
public:
    void SetEngineInterfaces(void* engine_server, void* player_info_mgr, void* server_game_dll);
    void SetGenericMap(const BspMapData* map_data);
    bool IsGenericMode() const { return generic_map_ != nullptr; }
    const BspMapData* GetGenericMap() const { return generic_map_; }
    void OnResetChamber(int chamber_id);
    void NotifyPortalPlaced(const std::string& color, const std::string& wall_id);
    void NotifyPortalPlacedWall(const std::string& color, const PortalWallRecord& wall);
    void NotifyHoldState(const std::string& held_entity_id);
    void NotifyPedestalPressed(float duration);

    BlackboardState Extract(int chamber_id, int tick, float dt);
    void SetPlayerVelocityXY(float vx, float vy, bool zero_z = false);
    void ApplyBaseVelocityXY(float dvx, float dvy);
    void SetPlayerEyeAngles(float pitch, float yaw);
    static std::string ToJson(const BlackboardState& state);

private:
    BlackboardState ExtractGeneric(int tick, float dt);

    void* engine_ = nullptr;
    void* player_info_mgr_ = nullptr;
    void* server_game_dll_ = nullptr;
    const BspMapData* generic_map_ = nullptr;

    Vec3 prev_player_pos_{};
    bool has_prev_pos_ = false;
    float tracked_yaw_ = 0.0f;
    float tracked_pitch_ = 0.0f;
    bool has_tracked_yaw_ = false;
    int prev_zone_id_ = -1;
    std::string tracked_held_id_;
    float pedestal_timer_ = 0.0f;
    PortalRecord tracked_blue_{"blue"};
    PortalRecord tracked_orange_{"orange"};
    int netprop_portal_activated_ = -1;
    int netprop_portal_is2_ = -1;
    int netprop_player_vel_x_ = -1;
    int netprop_player_vel_y_ = -1;
    int netprop_player_vel_z_ = -1;
    int netprop_player_basevel_ = -1;
    int netprop_player_flags_ = -1;
    int netprop_eye_pitch_ = -1;
    int netprop_eye_yaw_ = -1;
    bool netprops_resolved_ = false;

    void ResolveNetProps();
};

}  // namespace portal_ml
