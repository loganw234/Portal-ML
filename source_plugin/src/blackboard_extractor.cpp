#include "blackboard_extractor.h"
#include "bsp_scanner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "bspflags.h"
#include "cmodel.h"
#include "const.h"
#include "eiface.h"
#include "edict.h"
#include "engine/ICollideable.h"
#include "engine/IEngineTrace.h"
#include "game/server/iplayerinfo.h"
#include "iserverunknown.h"
#include "server_class.h"
#include "dt_send.h"
#endif

namespace portal_ml {

namespace {

constexpr float kButtonTriggerRadius = 52.0f;
constexpr float kExitTriggerRadius = 40.0f;

float DistXY(const Vec3& a, const Vec3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

void WriteVec3(std::ostringstream& os, const Vec3& v) {
    os << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}

void WriteNullableString(std::ostringstream& os, const std::string& s) {
    if (s.empty()) {
        os << "null";
    } else {
        os << '"' << s << '"';
    }
}

void WritePortal(std::ostringstream& os, const PortalRecord& p) {
    os << "{\"color\":\"" << p.color << "\","
       << "\"active\":" << (p.active ? "true" : "false") << ','
       << "\"wall_id\":";
    WriteNullableString(os, p.wall_id);
    os << ",\"pos\":";
    WriteVec3(os, p.pos);
    os << ",\"normal\":";
    WriteVec3(os, p.normal);
    os << ",\"zone_id\":" << p.zone_id << '}';
}

#ifdef PORTAL_ML_USE_SOURCE_SDK
int FindPropInTable(SendTable* table, const char* prop_name, int base_offset = 0) {
    if (!table) return -1;
    const int n = table->GetNumProps();
    for (int i = 0; i < n; ++i) {
        SendProp* prop = table->GetProp(i);
        if (!prop) continue;
        const char* name = prop->GetName();
        if (name && std::strcmp(name, prop_name) == 0) {
            return base_offset + prop->GetOffset();
        }
        if (prop->GetType() == DPT_DataTable && prop->GetDataTable()) {
            const int found = FindPropInTable(prop->GetDataTable(), prop_name, base_offset + prop->GetOffset());
            if (found >= 0) return found;
        }
    }
    return -1;
}

int FindSendPropOffset(IServerGameDLL* dll, const char* class_name, const char* prop_name) {
    if (!dll) return -1;
    for (ServerClass* sc = dll->GetAllServerClasses(); sc != nullptr; sc = sc->m_pNext) {
        if (sc->m_pNetworkName && std::strcmp(sc->m_pNetworkName, class_name) == 0) {
            return FindPropInTable(sc->m_pTable, prop_name);
        }
    }
    return -1;
}
#endif

}  // namespace

const ChamberLayout& GetChamberLayout(int chamber_id) {
    // Exact 1:1 coordinates matching portal_ml/sim/chambers.py and portal_ml_5chambers.bsp
    static const ChamberLayout kLayouts[7] = {
        // Chamber 1: The Hallway
        {1, 0.0f, 256.0f, {64.0f, 128.0f, 8.0f}, 0.0f, true,
         false, "", {}, false, "", {}, false, "", {},
         "c1_exit_0", {480.0f, 128.0f, 0.0f}, 120,
         {{0, 0.0f, 512.0f, 0.0f, 256.0f}}},
        // Chamber 2: The Pedestal
        {2, 2048.0f, 2304.0f, {64.0f, 2176.0f, 8.0f}, 0.0f, false,
         false, "", {}, false, "", {}, true, "c2_pedestal_0", {224.0f, 2112.0f, 0.0f},
         "c2_exit_0", {480.0f, 2176.0f, 0.0f}, 160,
         {{0, 0.0f, 512.0f, 2048.0f, 2304.0f}}},
        // Chamber 3: Box on Button
        {3, 4096.0f, 4480.0f, {64.0f, 4288.0f, 8.0f}, 0.0f, false,
         true, "c3_cube_0", {192.0f, 4160.0f, 24.0f},
         true, "c3_floor_btn_0", {320.0f, 4400.0f, 0.0f},
         false, "", {},
         "c3_exit_0", {480.0f, 4288.0f, 0.0f}, 200,
         {{0, 0.0f, 512.0f, 4096.0f, 4480.0f}}},
        // Chamber 4: The Simple Gap
        {4, 6144.0f, 6400.0f, {96.0f, 6272.0f, 8.0f}, 0.0f, true,
         false, "", {}, false, "", {}, false, "", {},
         "c4_exit_0", {608.0f, 6272.0f, 0.0f}, 180,
         {{0, 0.0f, 256.0f, 6144.0f, 6400.0f},
          {1, 384.0f, 640.0f, 6144.0f, 6400.0f}}},
        // Chamber 5: Held-Out Generalization Test (rotated 90 deg)
        {5, 8192.0f, 8832.0f, {192.0f, 8272.0f, 8.0f}, 90.0f, false,
         true, "c5_cube_0", {192.0f, 8720.0f, 24.0f},
         true, "c5_floor_btn_0", {304.0f, 8352.0f, 0.0f},
         false, "", {},
         "c5_exit_0", {96.0f, 8224.0f, 0.0f}, 260,
         {{0, 0.0f, 384.0f, 8192.0f, 8448.0f},
          {1, 0.0f, 384.0f, 8576.0f, 8832.0f}}},
        // Chamber 6: Held-Out Generalization Test — Button Across Gap
        {6, 10240.0f, 10624.0f, {96.0f, 10432.0f, 8.0f}, 0.0f, false,
         true, "c6_cube_0", {176.0f, 10304.0f, 24.0f},
         true, "c6_floor_btn_0", {528.0f, 10432.0f, 0.0f},
         false, "", {},
         "c6_exit_0", {48.0f, 10560.0f, 0.0f}, 260,
         {{0, 0.0f, 256.0f, 10240.0f, 10624.0f},
          {1, 384.0f, 640.0f, 10240.0f, 10624.0f}}},
        // Chamber 7: Held-Out Generalization Test — Three-Zone Two-Gap Relay
        {7, 12288.0f, 12544.0f, {128.0f, 12416.0f, 8.0f}, 0.0f, false,
         true, "c7_cube_0", {912.0f, 12416.0f, 24.0f},
         true, "c7_floor_btn_0", {512.0f, 12416.0f, 0.0f},
         false, "", {},
         "c7_exit_0", {48.0f, 12416.0f, 0.0f}, 320,
         {{0, 0.0f, 256.0f, 12288.0f, 12544.0f},
          {1, 384.0f, 640.0f, 12288.0f, 12544.0f},
          {2, 768.0f, 1024.0f, 12288.0f, 12544.0f}}},
    };
    const int idx = (chamber_id >= 1 && chamber_id <= 7) ? (chamber_id - 1) : 0;
    return kLayouts[idx];
}

std::vector<PortalWallRecord> GetChamberPortalWalls(int chamber_id) {
    if (chamber_id == 4) {
        return {
            {"c4_wall_a", {128.0f, 6144.0f, 64.0f}, {0.0f, 1.0f, 0.0f}, 0, 128.0f, 128.0f},
            {"c4_wall_b", {512.0f, 6144.0f, 64.0f}, {0.0f, 1.0f, 0.0f}, 1, 128.0f, 128.0f},
        };
    }
    if (chamber_id == 5) {
        return {
            {"c5_wall_a", {0.0f, 8320.0f, 64.0f}, {1.0f, 0.0f, 0.0f}, 0, 128.0f, 128.0f},
            {"c5_wall_b", {384.0f, 8704.0f, 64.0f}, {-1.0f, 0.0f, 0.0f}, 1, 128.0f, 128.0f},
        };
    }
    if (chamber_id == 6) {
        return {
            {"c6_wall_a", {128.0f, 10240.0f, 64.0f}, {0.0f, 1.0f, 0.0f}, 0, 128.0f, 128.0f},
            {"c6_wall_b", {512.0f, 10624.0f, 64.0f}, {0.0f, -1.0f, 0.0f}, 1, 128.0f, 128.0f},
        };
    }
    if (chamber_id == 7) {
        return {
            {"c7_wall_a", {128.0f, 12288.0f, 64.0f}, {0.0f, 1.0f, 0.0f}, 0, 128.0f, 128.0f},
            {"c7_wall_b", {512.0f, 12288.0f, 64.0f}, {0.0f, 1.0f, 0.0f}, 1, 128.0f, 128.0f},
            {"c7_wall_c", {896.0f, 12288.0f, 64.0f}, {0.0f, 1.0f, 0.0f}, 2, 128.0f, 128.0f},
        };
    }
    return {};
}

int ClassifyZone(const ChamberLayout& layout, float x, float y) {
    for (const auto& z : layout.zones) {
        if (x >= z.min_x && x <= z.max_x && y >= z.min_y && y <= z.max_y) {
            return z.zone_id;
        }
    }
    // Nearest zone fallback
    int best = layout.zones.empty() ? 0 : layout.zones[0].zone_id;
    float best_d = 1e9f;
    for (const auto& z : layout.zones) {
        const float cx = 0.5f * (z.min_x + z.max_x);
        const float cy = 0.5f * (z.min_y + z.max_y);
        const float d = std::hypot(x - cx, y - cy);
        if (d < best_d) {
            best_d = d;
            best = z.zone_id;
        }
    }
    return best;
}

void BlackboardExtractor::SetEngineInterfaces(
    void* engine_server, void* player_info_mgr, void* server_game_dll, void* engine_trace) {
    engine_ = engine_server;
    player_info_mgr_ = player_info_mgr;
    server_game_dll_ = server_game_dll;
    engine_trace_ = engine_trace;
}

void BlackboardExtractor::SetGenericMap(const BspMapData* map_data) {
    generic_map_ = (map_data && map_data->loaded) ? map_data : nullptr;
}

void BlackboardExtractor::ResolveNetProps() {
    if (netprops_resolved_) return;
    netprops_resolved_ = true;
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* dll = static_cast<IServerGameDLL*>(server_game_dll_);
    if (dll) {
        netprop_portal_activated_ = FindSendPropOffset(dll, "CProp_Portal", "m_bActivated");
        netprop_portal_is2_ = FindSendPropOffset(dll, "CProp_Portal", "m_bIsPortal2");
        for (ServerClass* sc = dll->GetAllServerClasses(); sc != nullptr; sc = sc->m_pNext) {
            if (!sc->m_pNetworkName) continue;
            if (std::strstr(sc->m_pNetworkName, "Player") != nullptr) {
                const int ox = FindPropInTable(sc->m_pTable, "m_vecVelocity[0]");
                const int oy = FindPropInTable(sc->m_pTable, "m_vecVelocity[1]");
                const int oz = FindPropInTable(sc->m_pTable, "m_vecVelocity[2]");
                if (ox >= 0 && oy >= 0) {
                    netprop_player_vel_x_ = ox;
                    netprop_player_vel_y_ = oy;
                    netprop_player_vel_z_ = oz;
                    netprop_player_basevel_ = FindPropInTable(sc->m_pTable, "m_vecBaseVelocity");
                    netprop_player_flags_ = FindPropInTable(sc->m_pTable, "m_fFlags");
                    netprop_player_health_ = FindPropInTable(sc->m_pTable, "m_iHealth");
                    netprop_player_waterlevel_ = FindPropInTable(sc->m_pTable, "m_nWaterLevel");
                    netprop_ground_entity_ = FindPropInTable(sc->m_pTable, "m_hGroundEntity");
                    netprop_eye_pitch_ = FindPropInTable(sc->m_pTable, "m_angEyeAngles[0]");
                    netprop_eye_yaw_ = FindPropInTable(sc->m_pTable, "m_angEyeAngles[1]");
                    break;
                }
            }
        }
    }
#endif
}

void BlackboardExtractor::SetPlayerVelocityXY(float vx, float vy, bool zero_z) {
    ResolveNetProps();
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (!engine || netprop_player_vel_x_ < 0 || netprop_player_vel_y_ < 0) return;
    edict_t* player_ed = engine->PEntityOfEntIndex(1);
    if (!player_ed || player_ed->IsFree()) return;
    IServerUnknown* unk = player_ed->GetUnknown();
    if (!unk) return;
    auto* base = reinterpret_cast<uint8_t*>(unk->GetBaseEntity());
    if (!base) return;
    *reinterpret_cast<float*>(base + netprop_player_vel_x_) = vx;
    *reinterpret_cast<float*>(base + netprop_player_vel_y_) = vy;
    if (zero_z && netprop_player_vel_z_ >= 0) {
        *reinterpret_cast<float*>(base + netprop_player_vel_z_) = 0.0f;
    }
    if (vx == 0.0f && vy == 0.0f && netprop_player_basevel_ >= 0) {
        auto* bv = reinterpret_cast<float*>(base + netprop_player_basevel_);
        bv[0] = 0.0f;
        bv[1] = 0.0f;
        bv[2] = 0.0f;
        if (netprop_player_flags_ >= 0) {
            auto* flags = reinterpret_cast<int*>(base + netprop_player_flags_);
            *flags &= ~FL_BASEVELOCITY;
        }
    }
#else
    (void)vx;
    (void)vy;
    (void)zero_z;
#endif
}

void BlackboardExtractor::ApplyBaseVelocityXY(float dvx, float dvy) {
    ResolveNetProps();
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (!engine || netprop_player_basevel_ < 0 || netprop_player_flags_ < 0) return;
    edict_t* player_ed = engine->PEntityOfEntIndex(1);
    if (!player_ed || player_ed->IsFree()) return;
    IServerUnknown* unk = player_ed->GetUnknown();
    if (!unk) return;
    auto* base = reinterpret_cast<uint8_t*>(unk->GetBaseEntity());
    if (!base) return;
    auto* bv = reinterpret_cast<float*>(base + netprop_player_basevel_);
    // FL_BASEVELOCITY persists across sub-ticks in CGameMovement; 0.72x matches 240 u/s mock speed.
    bv[0] = dvx * 0.72f;
    bv[1] = dvy * 0.72f;
    bv[2] = 0.0f;
    auto* flags = reinterpret_cast<int*>(base + netprop_player_flags_);
    *flags |= FL_BASEVELOCITY;  // (1<<23) in Portal 1: consumed natively by CGameMovement
#else
    (void)dvx;
    (void)dvy;
#endif
}

void BlackboardExtractor::ApplyJumpImpulse(float vz) {
    ResolveNetProps();
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (!engine || netprop_player_vel_z_ < 0) return;
    edict_t* player_ed = engine->PEntityOfEntIndex(1);
    if (!player_ed || player_ed->IsFree()) return;
    IServerUnknown* unk = player_ed->GetUnknown();
    if (!unk) return;
    auto* base = reinterpret_cast<uint8_t*>(unk->GetBaseEntity());
    if (!base) return;
    *reinterpret_cast<float*>(base + netprop_player_vel_z_) = vz;
    if (netprop_player_basevel_ >= 0) {
        auto* bv = reinterpret_cast<float*>(base + netprop_player_basevel_);
        bv[2] = vz;
    }
    if (netprop_ground_entity_ >= 0) {
        *reinterpret_cast<int*>(base + netprop_ground_entity_) = -1;
    }
    if (netprop_player_flags_ >= 0) {
        auto* flags = reinterpret_cast<int*>(base + netprop_player_flags_);
        *flags &= ~FL_ONGROUND;
        *flags |= FL_BASEVELOCITY;
    }
#else
    (void)vz;
#endif
}

void BlackboardExtractor::ReadPlayerProprioception(BlackboardState& state) {
    ResolveNetProps();
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (!engine) return;
    edict_t* player_ed = engine->PEntityOfEntIndex(1);
    if (!player_ed || player_ed->IsFree()) return;
    IServerUnknown* unk = player_ed->GetUnknown();
    if (!unk) return;
    const auto* base = reinterpret_cast<const uint8_t*>(unk->GetBaseEntity());
    if (!base) return;
    if (netprop_player_health_ >= 0) {
        state.player_health = static_cast<float>(*reinterpret_cast<const int*>(base + netprop_player_health_));
    }
    if (netprop_player_flags_ >= 0) {
        const int flags = *reinterpret_cast<const int*>(base + netprop_player_flags_);
        state.player_on_ground = (flags & FL_ONGROUND) != 0;
        state.player_ducking = (flags & FL_DUCKING) != 0;
    }
    if (netprop_player_waterlevel_ >= 0) {
        state.player_water_level = static_cast<int>(*reinterpret_cast<const uint8_t*>(base + netprop_player_waterlevel_));
    }
#else
    (void)state;
#endif
}

void BlackboardExtractor::ComputeLocalSensors(BlackboardState& state) {
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* trace = static_cast<IEngineTrace*>(engine_trace_);
    if (!trace) return;

    IHandleEntity* player_handle = nullptr;
    if (auto* engine = static_cast<IVEngineServer*>(engine_)) {
        if (edict_t* ped = engine->PEntityOfEntIndex(1); ped && !ped->IsFree()) {
            player_handle = ped->GetUnknown();
        }
    }
    struct SkipPlayerFilter final : public CTraceFilter {
        IHandleEntity* skip = nullptr;
        bool ShouldHitEntity(IHandleEntity* ent, int /*mask*/) override { return ent != skip; }
    } filter;
    filter.skip = player_handle;
    const float yaw_rad = state.player_yaw * (3.14159265f / 180.0f);
    const float chest_z = state.player_pos.z + 36.0f;
    const float knee_z = state.player_pos.z + 18.0f;
    const float high_z = state.player_pos.z + 52.0f;
    constexpr float kRayRange = 256.0f;
    constexpr float kHazardRange = 192.0f;

    // 1. 8 horizontal obstacle rays at chest height
    static const float kRelAngles[8] = {0.0f, 45.0f, 90.0f, 135.0f, 180.0f, -135.0f, -90.0f, -45.0f};
    for (int i = 0; i < 8; ++i) {
        const float ang = yaw_rad + kRelAngles[i] * (3.14159265f / 180.0f);
        Vector start(state.player_pos.x, state.player_pos.y, chest_z);
        Vector end(start.x + std::cos(ang) * kRayRange, start.y + std::sin(ang) * kRayRange, chest_z);
        Ray_t ray;
        ray.Init(start, end);
        trace_t tr;
        trace->TraceRay(ray, MASK_PLAYERSOLID, &filter, &tr);
        state.sensors.wall_rays[i] = tr.fraction;
    }

    // 2. 3 downward floor-drop probes at 48u ahead ([0, +35, -35] deg)
    static const float kFloorAngles[3] = {0.0f, 35.0f, -35.0f};
    for (int i = 0; i < 3; ++i) {
        const float ang = yaw_rad + kFloorAngles[i] * (3.14159265f / 180.0f);
        const float px = state.player_pos.x + std::cos(ang) * 48.0f;
        const float py = state.player_pos.y + std::sin(ang) * 48.0f;
        Vector start(px, py, chest_z);
        Vector end(px, py, chest_z - 128.0f);
        Ray_t ray;
        ray.Init(start, end);
        trace_t tr;
        trace->TraceRay(ray, MASK_PLAYERSOLID, &filter, &tr);
        state.sensors.floor_probes[i] = tr.fraction;
    }

    // 3. Forward knee-vs-high hurdle clearance probe (72u ahead).
    // Uses world-and-static-props filter so movable prop_physics cubes do not
    // masquerade as jumpable hurdles when approaching a cube to pick it up.
    {
        CTraceFilterWorldAndPropsOnly static_filter;
        const float fdx = std::cos(yaw_rad) * 72.0f;
        const float fdy = std::sin(yaw_rad) * 72.0f;
        Ray_t r_knee, r_high;
        r_knee.Init(
            Vector(state.player_pos.x, state.player_pos.y, knee_z),
            Vector(state.player_pos.x + fdx, state.player_pos.y + fdy, knee_z));
        r_high.Init(
            Vector(state.player_pos.x, state.player_pos.y, high_z),
            Vector(state.player_pos.x + fdx, state.player_pos.y + fdy, high_z));
        trace_t tr_k, tr_h;
        trace->TraceRay(r_knee, MASK_PLAYERSOLID, &static_filter, &tr_k);
        trace->TraceRay(r_high, MASK_PLAYERSOLID, &static_filter, &tr_h);
        if (tr_k.fraction < 0.94f && tr_h.fraction >= 0.94f) {
            state.sensors.step_clearance = 1.0f;
        } else if (tr_k.fraction < 0.94f && tr_h.fraction < 0.94f) {
            state.sensors.step_clearance = -1.0f;
        } else {
            state.sensors.step_clearance = 0.0f;
        }
    }

    // 4. Forward liquid/slime + BSP trigger_hurt hazard proximity probe
    float best_hazard = 0.0f;
    {
        Vector start(state.player_pos.x, state.player_pos.y, state.player_pos.z + 12.0f);
        Vector end(
            start.x + std::cos(yaw_rad) * kHazardRange,
            start.y + std::sin(yaw_rad) * kHazardRange,
            state.player_pos.z - 48.0f);
        Ray_t ray;
        ray.Init(start, end);
        trace_t tr;
        trace->TraceRay(ray, CONTENTS_SLIME | CONTENTS_WATER, &filter, &tr);
        if (tr.fraction < 1.0f) {
            best_hazard = 1.0f - tr.fraction;
        }
    }
    const float fdx = std::cos(yaw_rad);
    const float fdy = std::sin(yaw_rad);
    for (const auto& h : state.hazards) {
        if (state.player_pos.x >= h.min_pos.x && state.player_pos.x <= h.max_pos.x &&
            state.player_pos.y >= h.min_pos.y && state.player_pos.y <= h.max_pos.y) {
            best_hazard = 1.0f;
            break;
        }
        // Sample along forward ray up to kHazardRange
        for (float d = 16.0f; d <= kHazardRange; d += 16.0f) {
            const float sx = state.player_pos.x + fdx * d;
            const float sy = state.player_pos.y + fdy * d;
            if (sx >= h.min_pos.x && sx <= h.max_pos.x && sy >= h.min_pos.y && sy <= h.max_pos.y) {
                const float prox = 1.0f - (d / kHazardRange);
                if (prox > best_hazard) best_hazard = prox;
                break;
            }
        }
    }
    state.sensors.hazard_ahead = best_hazard;
#else
    (void)state;
#endif
}

void BlackboardExtractor::SetPlayerEyeAngles(float pitch, float yaw) {
    tracked_pitch_ = pitch;
    tracked_yaw_ = yaw;
    has_tracked_yaw_ = true;
    ResolveNetProps();
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (!engine) return;
    edict_t* player_ed = engine->PEntityOfEntIndex(1);
    if (!player_ed || player_ed->IsFree()) return;
    std::ostringstream ss;
    ss << "setang " << pitch << ' ' << yaw << " 0\n";
    engine->ServerCommand(ss.str().c_str());
    IServerUnknown* unk = player_ed->GetUnknown();
    if (!unk) return;
    auto* base = reinterpret_cast<uint8_t*>(unk->GetBaseEntity());
    if (!base) return;
    if (netprop_eye_pitch_ >= 0) {
        *reinterpret_cast<float*>(base + netprop_eye_pitch_) = pitch;
    }
    if (netprop_eye_yaw_ >= 0) {
        *reinterpret_cast<float*>(base + netprop_eye_yaw_) = yaw;
    }
#else
    (void)pitch;
    (void)yaw;
#endif
}

void BlackboardExtractor::OnResetChamber(int chamber_id) {
    if (generic_map_) {
        active_room_spawn_ = generic_map_->player_spawn;
        active_room_yaw_ = generic_map_->player_spawn_yaw;
        if (chamber_id >= 1 &&
            static_cast<size_t>(chamber_id) <= generic_map_->room_spawns.size()) {
            const auto& sp = generic_map_->room_spawns[static_cast<size_t>(chamber_id - 1)];
            active_room_spawn_ = sp.pos;
            active_room_yaw_ = sp.yaw;
        }
        active_room_base_z_ = active_room_spawn_.z;
        prev_player_pos_ = active_room_spawn_;
        has_prev_pos_ = false;
        tracked_yaw_ = active_room_yaw_;
        tracked_pitch_ = 0.0f;
        has_tracked_yaw_ = false;  // Let real campaign maps use live player eye angles
        prev_zone_id_ = 0;
        tracked_held_id_.clear();
        pedestal_timer_ = 0.0f;
        tracked_blue_ = PortalRecord{"blue"};
        tracked_orange_ = PortalRecord{"orange"};
        return;
    }
    const auto& layout = GetChamberLayout(chamber_id);
    prev_player_pos_ = layout.player_spawn;
    has_prev_pos_ = false;
    tracked_yaw_ = layout.player_yaw;
    tracked_pitch_ = 0.0f;
    has_tracked_yaw_ = true;
    prev_zone_id_ = ClassifyZone(layout, layout.player_spawn.x, layout.player_spawn.y);
    tracked_held_id_.clear();
    pedestal_timer_ = 0.0f;
    tracked_blue_ = PortalRecord{"blue"};
    tracked_orange_ = PortalRecord{"orange"};
}

void BlackboardExtractor::NotifyPortalPlacedWall(const std::string& color, const PortalWallRecord& w) {
    PortalRecord& target = (color == "blue") ? tracked_blue_ : tracked_orange_;
    PortalRecord& other = (color == "blue") ? tracked_orange_ : tracked_blue_;
    if (other.active && other.wall_id == w.id) {
        other.active = false;
        other.wall_id.clear();
        other.zone_id = -1;
    }
    target.active = true;
    target.wall_id = w.id;
    target.pos = w.pos;
    target.normal = w.normal;
    target.zone_id = w.zone_id;
}

void BlackboardExtractor::NotifyPortalPlaced(const std::string& color, const std::string& wall_id) {
    if (generic_map_) {
        for (const auto& w : generic_map_->portal_surfaces) {
            if (w.id == wall_id) {
                NotifyPortalPlacedWall(color, w);
                return;
            }
        }
    }
    // Updated immediately on ShootPortal and cross-checked against live prop_portal edicts
    for (int cid = 4; cid <= 7; ++cid) {
        for (const auto& w : GetChamberPortalWalls(cid)) {
            if (w.id == wall_id) {
                NotifyPortalPlacedWall(color, w);
                return;
            }
        }
    }
}

void BlackboardExtractor::NotifyHoldState(const std::string& held_entity_id) {
    tracked_held_id_ = held_entity_id;
}

void BlackboardExtractor::NotifyPedestalPressed(float duration) {
    pedestal_timer_ = duration;
}

BlackboardState BlackboardExtractor::Extract(int chamber_id, int tick, float dt) {
    ResolveNetProps();
    if (generic_map_) {
        return ExtractGeneric(tick, dt);
    }
    const auto& layout = GetChamberLayout(chamber_id);

    BlackboardState state;
    state.tick = tick;
    state.dt = dt;
    state.chamber_id = chamber_id;
    state.player_pos = layout.player_spawn;
    state.player_yaw = layout.player_yaw;
    state.portal_walls = GetChamberPortalWalls(chamber_id);
    state.blue_portal = tracked_blue_;
    state.orange_portal = tracked_orange_;

    if (pedestal_timer_ > 0.0f) {
        pedestal_timer_ = (std::max)(0.0f, pedestal_timer_ - dt);
    }

    Vec3 live_cube_pos = layout.cube_spawn;
    bool live_door_open = layout.exit_starts_open;

#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    auto* pim = static_cast<IPlayerInfoManager*>(player_info_mgr_);

    if (engine && pim) {
        edict_t* player_ed = engine->PEntityOfEntIndex(1);
        if (player_ed && !player_ed->IsFree()) {
            IPlayerInfo* pinfo = pim->GetPlayerInfo(player_ed);
            if (pinfo && pinfo->IsConnected()) {
                const Vector org = pinfo->GetAbsOrigin();
                const QAngle ang = pinfo->GetAbsAngles();
                state.player_pos = {org.x, org.y, org.z};
                state.player_yaw = ang.y;
                state.player_pitch = ang.x;
                // Prefer live server m_angEyeAngles over client-replicated GetAbsAngles
                // so yaw updates deterministically even when hl2.exe runs unfocused/fast.
                IServerUnknown* punk = player_ed->GetUnknown();
                if (punk) {
                    const auto* pbase = reinterpret_cast<const uint8_t*>(punk->GetBaseEntity());
                    if (pbase && netprop_eye_yaw_ >= 0) {
                        state.player_yaw = *reinterpret_cast<const float*>(pbase + netprop_eye_yaw_);
                    }
                    if (pbase && netprop_eye_pitch_ >= 0) {
                        state.player_pitch = *reinterpret_cast<const float*>(pbase + netprop_eye_pitch_);
                    }
                }
            }
        }

        // Scan live server edicts inside the active chamber's Y-band
        const int max_ents = engine->GetEntityCount();
        for (int i = 2; i < max_ents; ++i) {
            edict_t* ed = engine->PEntityOfEntIndex(i);
            if (!ed || ed->IsFree()) continue;
            const char* cls = ed->GetClassName();
            if (!cls) continue;
            IServerUnknown* unk = ed->GetUnknown();
            if (!unk) continue;
            ICollideable* col = unk->GetCollideable();
            if (!col) continue;
            const Vector pos = col->GetCollisionOrigin();

            const bool in_chamber_y = (pos.y >= layout.min_y - 64.0f && pos.y <= layout.max_y + 64.0f);
            if (!in_chamber_y) continue;

            if (layout.has_cube && std::strcmp(cls, "prop_physics") == 0) {
                live_cube_pos = {pos.x, pos.y, pos.z};
            } else if (std::strcmp(cls, "func_door") == 0) {
                // Sliding door opens upward (movedir -90 0 0): closed z~56, open z>96
                live_door_open = (pos.z > 88.0f);
            } else if (std::strcmp(cls, "prop_portal") == 0 &&
                       netprop_portal_activated_ >= 0 && netprop_portal_is2_ >= 0) {
                const auto* base = reinterpret_cast<const uint8_t*>(unk->GetBaseEntity());
                if (base) {
                    const bool activated = *reinterpret_cast<const bool*>(base + netprop_portal_activated_);
                    const bool is_orange = *reinterpret_cast<const bool*>(base + netprop_portal_is2_);
                    if (activated) {
                        PortalRecord& pr = is_orange ? state.orange_portal : state.blue_portal;
                        pr.active = true;
                        pr.pos = {pos.x, pos.y, pos.z};
                        pr.zone_id = ClassifyZone(layout, pos.x, pos.y);
                    }
                }
            }
        }
    }
#endif

    if (has_prev_pos_ && dt > 1e-5f) {
        state.player_vel = {
            (state.player_pos.x - prev_player_pos_.x) / dt,
            (state.player_pos.y - prev_player_pos_.y) / dt,
            (state.player_pos.z - prev_player_pos_.z) / dt,
        };
    }
    prev_player_pos_ = state.player_pos;
    has_prev_pos_ = true;

    state.player_zone_id = ClassifyZone(layout, state.player_pos.x, state.player_pos.y);
    if (has_tracked_yaw_) {
        if (prev_zone_id_ >= 0 && state.player_zone_id != prev_zone_id_) {
            // Re-sync orientation after a portal teleport
            tracked_yaw_ = state.player_yaw;
            tracked_pitch_ = state.player_pitch;
        } else {
            // Avoid client CUserCmd echo overwriting server-commanded setang when unfocused
            state.player_yaw = tracked_yaw_;
            state.player_pitch = tracked_pitch_;
        }
    }
    prev_zone_id_ = state.player_zone_id;
    state.held_entity_id = tracked_held_id_;

    bool floor_btn_pressed = false;
    if (layout.has_cube) {
        const bool is_held = (!tracked_held_id_.empty() && tracked_held_id_ == layout.cube_id);
        const int cube_zone = is_held ? state.player_zone_id
                                      : ClassifyZone(layout, live_cube_pos.x, live_cube_pos.y);
        bool on_btn = false;
        if (layout.has_floor_button && !is_held &&
            cube_zone == ClassifyZone(layout, layout.floor_button_pos.x, layout.floor_button_pos.y) &&
            DistXY(live_cube_pos, layout.floor_button_pos) <= kButtonTriggerRadius) {
            on_btn = true;
            floor_btn_pressed = true;
        }
        state.entities.push_back({
            layout.cube_id, "cube", live_cube_pos, {}, cube_zone, on_btn, is_held, 0.0f
        });
    }

    if (layout.has_floor_button) {
        const int btn_zone = ClassifyZone(layout, layout.floor_button_pos.x, layout.floor_button_pos.y);
        state.entities.push_back({
            layout.floor_button_id, "floor_button", layout.floor_button_pos, {},
            btn_zone, floor_btn_pressed, false, 0.0f
        });
    }

    if (layout.has_pedestal) {
        const int ped_zone = ClassifyZone(layout, layout.pedestal_pos.x, layout.pedestal_pos.y);
        state.entities.push_back({
            layout.pedestal_id, "pedestal_button", layout.pedestal_pos, {},
            ped_zone, pedestal_timer_ > 0.0f, false, pedestal_timer_
        });
    }

    // In floor-button chambers, require the cube to be holding the button (not just the player stepping on its edge)
    const bool exit_open = layout.exit_starts_open ||
                           (layout.has_floor_button ? floor_btn_pressed
                                                    : (live_door_open || (pedestal_timer_ > 0.0f)));
    const int exit_zone = ClassifyZone(layout, layout.exit_pos.x, layout.exit_pos.y);
    state.entities.push_back({
        layout.exit_id, "exit_door", layout.exit_pos, {}, exit_zone, exit_open, false, 0.0f
    });

    if (exit_open && state.player_zone_id == exit_zone &&
        DistXY(state.player_pos, layout.exit_pos) <= kExitTriggerRadius) {
        state.done = true;
        state.success = true;
    } else if (tick >= layout.max_steps) {
        state.done = true;
        state.success = false;
    }

    ReadPlayerProprioception(state);
    ComputeLocalSensors(state);
    return state;
}

BlackboardState BlackboardExtractor::ExtractGeneric(int tick, float dt) {
    BlackboardState state;
    state.tick = tick;
    state.dt = dt;
    state.chamber_id = 0;  // 0 denotes generic live-BSP campaign/custom map
    if (active_room_spawn_.x == 0.0f && active_room_spawn_.y == 0.0f && active_room_spawn_.z == 0.0f) {
        active_room_spawn_ = generic_map_->player_spawn;
        active_room_base_z_ = generic_map_->player_spawn.z;
        active_room_yaw_ = generic_map_->player_spawn_yaw;
    }
    state.player_pos = active_room_spawn_;
    state.player_yaw = active_room_yaw_;
    state.blue_portal = tracked_blue_;
    state.orange_portal = tracked_orange_;

    std::vector<Vec3> live_props;
    std::vector<Vec3> live_doors;
    std::vector<Vec3> live_door_spawns;

#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    auto* pim = static_cast<IPlayerInfoManager*>(player_info_mgr_);
    if (engine && pim) {
        edict_t* player_ed = engine->PEntityOfEntIndex(1);
        if (player_ed && !player_ed->IsFree()) {
            IPlayerInfo* pinfo = pim->GetPlayerInfo(player_ed);
            if (pinfo && pinfo->IsConnected()) {
                const Vector org = pinfo->GetAbsOrigin();
                const QAngle ang = pinfo->GetAbsAngles();
                state.player_pos = {org.x, org.y, org.z};
                state.player_yaw = ang.y;
                state.player_pitch = ang.x;
            }
        }
        // Automatically follow multi-story elevator transitions to another room band
        for (const auto& sp : generic_map_->room_spawns) {
            if (std::fabs(state.player_pos.z - sp.pos.z) < 96.0f &&
                std::fabs(sp.pos.z - active_room_base_z_) > 220.0f) {
                active_room_base_z_ = sp.pos.z;
                active_room_spawn_ = sp.pos;
                active_room_yaw_ = sp.yaw;
                break;
            }
        }
    }
#endif

    const float room_base_z = active_room_base_z_;
    auto classify_generic_zone = [room_base_z](float z) -> int {
        if (z <= room_base_z - 160.0f) return 1; // deep impassable pit (e.g. Room 1 -256u)
        if (z >= room_base_z + 260.0f) return 0; // ceiling dropper chute -> approach in zone 0
        if (z >= room_base_z + 96.0f) return 2;  // upper balcony / ledge
        return 0;                                // main floor (incl. walkable stairs)
    };

#ifdef PORTAL_ML_USE_SOURCE_SDK
    if (engine && pim) {
        const int max_ents = engine->GetEntityCount();
        for (int i = 2; i < max_ents; ++i) {
            edict_t* ed = engine->PEntityOfEntIndex(i);
            if (!ed || ed->IsFree()) continue;
            const char* cls = ed->GetClassName();
            if (!cls) continue;
            IServerUnknown* unk = ed->GetUnknown();
            if (!unk) continue;
            ICollideable* col = unk->GetCollideable();
            if (!col) continue;
            const Vector pos = col->GetCollisionOrigin();

            if (std::strcmp(cls, "prop_physics") == 0) {
                live_props.push_back({pos.x, pos.y, pos.z});
            } else if (std::strcmp(cls, "func_door") == 0) {
                live_doors.push_back({pos.x, pos.y, pos.z});
            } else if (std::strcmp(cls, "prop_portal") == 0 &&
                       netprop_portal_activated_ >= 0 && netprop_portal_is2_ >= 0) {
                const auto* base = reinterpret_cast<const uint8_t*>(unk->GetBaseEntity());
                if (base) {
                    const bool activated = *reinterpret_cast<const bool*>(base + netprop_portal_activated_);
                    const bool is_orange = *reinterpret_cast<const bool*>(base + netprop_portal_is2_);
                    if (activated && pos.z >= room_base_z - 320.0f && pos.z <= room_base_z + 260.0f) {
                        const QAngle pang = col->GetCollisionAngles();
                        const float py = pang.y * (3.14159265f / 180.0f);
                        const float pp = pang.x * (3.14159265f / 180.0f);
                        PortalRecord& pr = is_orange ? state.orange_portal : state.blue_portal;
                        pr.active = true;
                        pr.pos = {pos.x, pos.y, pos.z};
                        pr.normal = {
                            std::cos(py) * std::cos(pp),
                            std::sin(py) * std::cos(pp),
                            -std::sin(pp),
                        };
                        pr.zone_id = classify_generic_zone(pos.z);
                        if (pr.wall_id.empty()) {
                            pr.wall_id = is_orange ? "live_portal_orange" : "live_portal_blue";
                        }
                    }
                }
            }
        }
    }
#endif

    if (has_prev_pos_ && dt > 1e-5f) {
        state.player_vel = {
            (state.player_pos.x - prev_player_pos_.x) / dt,
            (state.player_pos.y - prev_player_pos_.y) / dt,
            (state.player_pos.z - prev_player_pos_.z) / dt,
        };
    }
    prev_player_pos_ = state.player_pos;
    has_prev_pos_ = true;
    state.player_zone_id = classify_generic_zone(state.player_pos.z);
    if (has_tracked_yaw_) {
        if (prev_zone_id_ >= 0 && state.player_zone_id != prev_zone_id_) {
            tracked_yaw_ = state.player_yaw;
            tracked_pitch_ = state.player_pitch;
        } else {
            state.player_yaw = tracked_yaw_;
            state.player_pitch = tracked_pitch_;
        }
    }
    prev_zone_id_ = state.player_zone_id;
    state.held_entity_id = tracked_held_id_;

    auto dist3 = [](const Vec3& a, const Vec3& b) -> float {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Locate room-local elevator (for campaign exit progression after exit door opens)
    bool has_room_elevator = false;
    Vec3 room_elevator_pos{};
    float best_elev_d = 1e9f;
    for (const auto& de : generic_map_->entities) {
        if (de.kind == "elevator" &&
            de.spawn_pos.z >= room_base_z - 64.0f && de.spawn_pos.z <= room_base_z + 240.0f) {
            const float d = DistXY(de.spawn_pos, active_room_spawn_);
            if (d > 256.0f && d < best_elev_d) {
                best_elev_d = d;
                room_elevator_pos = de.spawn_pos;
                has_room_elevator = true;
            }
        }
    }

    // Count floor buttons and compute gameplay centroid in this story so lower-story
    // dropper templates at similar Z are never confused with active room cubes.
    int story_floor_btn_count = 0;
    Vec3 story_centroid{};
    int story_anchor_count = 0;
    for (const auto& de : generic_map_->entities) {
        if ((de.kind == "floor_button" || de.kind == "exit_door") &&
            de.spawn_pos.z >= room_base_z - 320.0f && de.spawn_pos.z <= room_base_z + 260.0f) {
            if (de.kind == "floor_button") ++story_floor_btn_count;
            story_centroid.x += de.spawn_pos.x;
            story_centroid.y += de.spawn_pos.y;
            story_centroid.z += de.spawn_pos.z;
            ++story_anchor_count;
        }
    }
    if (story_anchor_count > 0) {
        story_centroid.x /= static_cast<float>(story_anchor_count);
        story_centroid.y /= static_cast<float>(story_anchor_count);
        story_centroid.z /= static_cast<float>(story_anchor_count);
    } else {
        story_centroid = active_room_spawn_;
    }

    // Filter BSP-discovered gameplay entities to the active vertical room band
    std::vector<Vec3> cube_positions;
    bool any_room_button_active = false;
    for (const auto& de : generic_map_->entities) {
        if (de.kind == "elevator" || de.kind == "fizzler") continue;
        // Room Z-band filter: keep entities in this story (plus dropper cubes hovering above)
        const bool in_room_z =
            (de.spawn_pos.z >= room_base_z - 320.0f && de.spawn_pos.z <= room_base_z + 260.0f);
        const bool is_dropper_cube =
            (de.kind == "cube" && de.spawn_pos.z > room_base_z + 260.0f &&
             DistXY(de.spawn_pos, active_room_spawn_) < 1200.0f);
        if (!in_room_z && !is_dropper_cube) continue;

        if (de.kind == "cube" && story_floor_btn_count >= 2) {
            // Multi-button story (e.g. testchmb_a_02 Room 2): ignore lower-story dropper
            // templates outside this room's XY footprint, and expose every distinct live
            // in-room prop_physics cube so Layer 4 can sequence multiple cubes across buttons.
            if (DistXY(de.spawn_pos, story_centroid) > 700.0f) continue;
            std::vector<Vec3> story_cubes;
            for (const auto& lp : live_props) {
                if (lp.z < room_base_z - 220.0f || lp.z > room_base_z + 220.0f) continue;
                if (DistXY(lp, story_centroid) > 700.0f) continue;
                // Deduplicate co-located props within 24u
                bool dup = false;
                for (const auto& ex : story_cubes) {
                    if (dist3(ex, lp) < 24.0f) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) story_cubes.push_back(lp);
            }
            if (story_cubes.empty()) continue;  // lower-story dropper template, not in this room
            for (size_t ci = 0; ci < story_cubes.size(); ++ci) {
                const std::string cid =
                    (story_cubes.size() == 1) ? de.id : (de.id + "_" + std::to_string(ci));
                const bool is_held = (!tracked_held_id_.empty() && tracked_held_id_ == cid);
                const Vec3 cpos = is_held ? state.player_pos : story_cubes[ci];
                const int cz = is_held ? state.player_zone_id : classify_generic_zone(cpos.z);
                if (!is_held) cube_positions.push_back(cpos);
                state.entities.push_back({cid, "cube", cpos, {}, cz, false, is_held, 0.0f});
            }
            continue;
        }

        Vec3 pos = de.spawn_pos;
        bool active = false;
        const bool is_held = (!tracked_held_id_.empty() && tracked_held_id_ == de.id);

        if (de.kind == "cube") {
            // Prefer a live prop_physics already dropped into the playable chamber
            // (z <= room_base_z + 200) over a spare cube waiting inside a ceiling dropper.
            bool found_in_room = false;
            float best_room_d = 900.0f;
            float best_hopper_d = 384.0f;
            for (const auto& lp : live_props) {
                if (lp.z < room_base_z - 320.0f || lp.z > room_base_z + 640.0f) continue;
                const float d = DistXY(lp, de.spawn_pos);
                if (lp.z <= room_base_z + 140.0f) {
                    if (d < best_room_d) {
                        best_room_d = d;
                        pos = lp;
                        found_in_room = true;
                    }
                } else if (!found_in_room && d < best_hopper_d) {
                    best_hopper_d = d;
                    pos = lp;
                }
            }
            if (is_held) {
                pos = state.player_pos;
            } else {
                cube_positions.push_back(pos);
            }
        } else if (de.kind == "exit_door") {
            // Check whether physical sliding func_door moved away from closed doorway center
            // (paired doors rest ~34-36u from doorway center when shut and ~86u when open).
            float best_d = 160.0f;
            for (const auto& ld : live_doors) {
                const float d = dist3(ld, de.spawn_pos);
                if (d < best_d) {
                    best_d = d;
                    active = (d > 52.0f);
                }
            }
        }

        const int ent_zone = is_held ? state.player_zone_id : classify_generic_zone(pos.z);
        // When a cube is still up in a ceiling dropper, project its target slightly outside pit edge
        // so walking toward it triggers the dropper without falling prematurely.
        Vec3 exposed_pos = pos;
        if (de.kind == "cube" && !is_held && pos.z > room_base_z + 140.0f) {
            exposed_pos.y = pos.y + 64.0f;
            exposed_pos.z = room_base_z;
        }
        state.entities.push_back({de.id, de.kind, exposed_pos, {}, ent_zone, active, is_held, 0.0f});
    }

    // Check whether any physical room exit_door has slid open in Source Engine
    bool any_door_physically_open = false;
    bool has_any_door = false;
    int room_btn_total = 0;
    int room_btn_active = 0;
    for (const auto& e : state.entities) {
        if (e.kind == "exit_door") {
            has_any_door = true;
            if (e.active) any_door_physically_open = true;
        } else if (e.kind == "floor_button") {
            ++room_btn_total;
        }
    }

    // Activate floor buttons by ANY live in-room prop_physics cube (not held).
    // In single-button rooms with a physical func_door, require it to be physically open;
    // in multi-button rooms (e.g. testchmb_a_02 Room 2), track each button independently
    // and mark each cube sitting on a button as active=true so Layer 4 sequences cleanly.
    std::vector<Vec3> pressed_cube_positions;
    for (auto& e : state.entities) {
        if (e.kind == "floor_button") {
            for (const auto& lp : live_props) {
                if (!tracked_held_id_.empty() && DistXY(lp, state.player_pos) < 32.0f &&
                    std::fabs(lp.z - (state.player_pos.z + 108.0f)) < 48.0f) {
                    continue;  // ignore cube currently carried overhead
                }
                const float dist_xy = DistXY(lp, e.pos);
                const bool within_pad = (dist_xy <= 52.0f && std::fabs(lp.z - e.pos.z) <= 56.0f);
                const bool confirmed = (room_btn_total >= 2)
                                           ? (dist_xy <= 46.0f && std::fabs(lp.z - e.pos.z) <= 56.0f)
                                           : (has_any_door ? (within_pad && any_door_physically_open)
                                                           : (dist_xy <= kButtonTriggerRadius &&
                                                              std::fabs(lp.z - e.pos.z) <= 56.0f));
                if (confirmed) {
                    e.active = true;
                    any_room_button_active = true;
                    ++room_btn_active;
                    pressed_cube_positions.push_back(lp);
                    break;
                }
            }
        }
    }
    for (const auto& pcp : pressed_cube_positions) {
        // Match the nearest non-held, not-yet-active cube entity to this pressed pad
        int best_idx = -1;
        float best_d = 1e9f;
        for (size_t i = 0; i < state.entities.size(); ++i) {
            const auto& e = state.entities[i];
            if (e.kind == "cube" && !e.held && !e.active) {
                const float d = DistXY(e.pos, pcp);
                if (d < best_d) {
                    best_d = d;
                    best_idx = static_cast<int>(i);
                }
            }
        }
        if (best_idx >= 0) {
            auto& ce = state.entities[static_cast<size_t>(best_idx)];
            ce.active = true;
            ce.pos = pcp;
            ce.zone_id = classify_generic_zone(pcp.z);
        }
    }

    // Sync exit door active state and guide through open doorway -> room elevator
    const bool all_room_buttons_active = (room_btn_total > 0 && room_btn_active >= room_btn_total);
    for (auto& e : state.entities) {
        if (e.kind == "exit_door") {
            if (room_btn_total >= 2) {
                e.active = (all_room_buttons_active && any_door_physically_open);
            } else if (room_btn_total == 1) {
                // Require the cube (not just the player standing on it) to hold the button
                e.active = any_room_button_active;
            }
            if (e.active && has_room_elevator) {
                // First steer through open doorway; once past doorway, steer into exit elevator
                const bool past_door =
                    (DistXY(state.player_pos, room_elevator_pos) <
                     DistXY(e.pos, room_elevator_pos) + 16.0f);
                if (past_door) {
                    e.pos = {room_elevator_pos.x, room_elevator_pos.y, room_base_z};
                }
                if (DistXY(state.player_pos, room_elevator_pos) <= 72.0f) {
                    state.done = true;
                    state.success = true;
                }
            } else if (e.active && DistXY(state.player_pos, e.pos) <= kExitTriggerRadius) {
                state.done = true;
                state.success = true;
            }
        }
    }

    // Select walkable vertical portalable concrete walls per zone (adjusted to walkable portal Z)
    auto add_zone_walls = [&](int target_zone, float floor_z, size_t max_count) {
        std::vector<std::pair<float, size_t>> ranked;
        for (size_t i = 0; i < generic_map_->portal_surfaces.size(); ++i) {
            const auto& s = generic_map_->portal_surfaces[i];
            if (std::fabs(s.normal.z) > 0.25f) continue;  // vertical walls only
            if (s.pos.z < floor_z - 16.0f || s.pos.z > floor_z + 128.0f) continue;
            if (DistXY(s.pos, state.player_pos) > 1400.0f) continue;
            ranked.emplace_back(DistXY(state.player_pos, s.pos), i);
        }
        std::sort(ranked.begin(), ranked.end());
        if (ranked.size() > max_count) ranked.resize(max_count);
        for (const auto& item : ranked) {
            PortalWallRecord w = generic_map_->portal_surfaces[item.second];
            w.zone_id = target_zone;
            w.pos.z = floor_z + 56.0f;  // walkable eye/chest height above zone floor
            state.portal_walls.push_back(w);
        }
    };
    add_zone_walls(0, room_base_z, 8);
    add_zone_walls(1, room_base_z - 256.0f, 8);
    add_zone_walls(2, room_base_z + 128.0f, 8);

    // Ensure any live pre-placed Valve portal (e.g. portal_a_lvl3) has a matching PortalWall entry
    auto ensure_portal_wall = [&](const PortalRecord& p) {
        if (!p.active || p.wall_id.empty()) return;
        for (const auto& w : state.portal_walls) {
            if (w.id == p.wall_id) return;
        }
        state.portal_walls.push_back({p.wall_id, p.pos, p.normal, p.zone_id, 128.0f, 128.0f});
    };
    ensure_portal_wall(state.blue_portal);
    ensure_portal_wall(state.orange_portal);

    if (generic_map_) {
        for (const auto& hz : generic_map_->hazards) {
            state.hazards.push_back({hz.id, hz.kind, hz.min_pos, hz.max_pos, hz.dps});
        }
    }

    if (!state.done && tick >= 400) {
        state.done = true;
        state.success = false;
    }

    ReadPlayerProprioception(state);
    ComputeLocalSensors(state);
    return state;
}

std::string BlackboardExtractor::ToJson(const BlackboardState& s) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);

    os << "{\"tick\":" << s.tick
       << ",\"dt\":" << s.dt
       << ",\"chamber_id\":" << s.chamber_id
       << ",\"player\":{\"pos\":";
    WriteVec3(os, s.player_pos);
    os << ",\"vel\":";
    WriteVec3(os, s.player_vel);
    os << ",\"yaw\":" << s.player_yaw
       << ",\"pitch\":" << s.player_pitch
       << ",\"zone_id\":" << s.player_zone_id
       << ",\"held_entity_id\":";
    WriteNullableString(os, s.held_entity_id);
    os << ",\"health\":" << s.player_health
       << ",\"on_ground\":" << (s.player_on_ground ? "true" : "false")
       << ",\"ducking\":" << (s.player_ducking ? "true" : "false")
       << ",\"water_level\":" << s.player_water_level
       << "},\"entities\":[";

    for (size_t i = 0; i < s.entities.size(); ++i) {
        const auto& e = s.entities[i];
        if (i > 0) os << ',';
        os << "{\"id\":\"" << e.id
           << "\",\"kind\":\"" << e.kind
           << "\",\"pos\":";
        WriteVec3(os, e.pos);
        os << ",\"vel\":";
        WriteVec3(os, e.vel);
        os << ",\"zone_id\":" << e.zone_id
           << ",\"active\":" << (e.active ? "true" : "false")
           << ",\"held\":" << (e.held ? "true" : "false")
           << ",\"timer\":" << e.timer << '}';
    }

    os << "],\"portal_walls\":[";
    for (size_t i = 0; i < s.portal_walls.size(); ++i) {
        const auto& w = s.portal_walls[i];
        if (i > 0) os << ',';
        os << "{\"id\":\"" << w.id << "\",\"pos\":";
        WriteVec3(os, w.pos);
        os << ",\"normal\":";
        WriteVec3(os, w.normal);
        os << ",\"zone_id\":" << w.zone_id
           << ",\"size\":[" << w.width << ',' << w.height << "]}";
    }

    os << "],\"portals\":{\"blue\":";
    WritePortal(os, s.blue_portal);
    os << ",\"orange\":";
    WritePortal(os, s.orange_portal);
    os << "},\"sensors\":{\"wall_rays\":[";
    for (int i = 0; i < 8; ++i) {
        if (i > 0) os << ',';
        os << s.sensors.wall_rays[i];
    }
    os << "],\"floor_probes\":[";
    for (int i = 0; i < 3; ++i) {
        if (i > 0) os << ',';
        os << s.sensors.floor_probes[i];
    }
    os << "],\"step_clearance\":" << s.sensors.step_clearance
       << ",\"hazard_ahead\":" << s.sensors.hazard_ahead
       << "},\"hazards\":[";
    for (size_t i = 0; i < s.hazards.size(); ++i) {
        const auto& hz = s.hazards[i];
        if (i > 0) os << ',';
        os << "{\"id\":\"" << hz.id << "\",\"kind\":\"" << hz.kind << "\",\"min\":";
        WriteVec3(os, hz.min_pos);
        os << ",\"max\":";
        WriteVec3(os, hz.max_pos);
        os << ",\"dps\":" << hz.dps << '}';
    }
    os << "],\"done\":" << (s.done ? "true" : "false")
       << ",\"success\":" << (s.success ? "true" : "false") << '}';

    return os.str();
}

}  // namespace portal_ml
