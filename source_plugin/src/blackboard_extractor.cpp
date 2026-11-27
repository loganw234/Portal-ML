#include "blackboard_extractor.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
#include "edict.h"
#include "engine/ICollideable.h"
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
    void* engine_server, void* player_info_mgr, void* server_game_dll) {
    engine_ = engine_server;
    player_info_mgr_ = player_info_mgr;
    server_game_dll_ = server_game_dll;
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
    }
#else
    (void)vx;
    (void)vy;
    (void)zero_z;
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

void BlackboardExtractor::NotifyPortalPlaced(const std::string& color, const std::string& wall_id) {
    // Updated immediately on ShootPortal and cross-checked against live prop_portal edicts
    for (int cid = 4; cid <= 7; ++cid) {
        for (const auto& w : GetChamberPortalWalls(cid)) {
            if (w.id == wall_id) {
                PortalRecord& target = (color == "blue") ? tracked_blue_ : tracked_orange_;
                PortalRecord& other = (color == "blue") ? tracked_orange_ : tracked_blue_;
                if (other.active && other.wall_id == wall_id) {
                    other.active = false;
                    other.wall_id.clear();
                    other.zone_id = -1;
                }
                target.active = true;
                target.wall_id = w.id;
                target.pos = w.pos;
                target.normal = w.normal;
                target.zone_id = w.zone_id;
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
        pedestal_timer_ = std::max(0.0f, pedestal_timer_ - dt);
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
    os << "},\"entities\":[";

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
    os << "},\"done\":" << (s.done ? "true" : "false")
       << ",\"success\":" << (s.success ? "true" : "false") << '}';

    return os.str();
}

}  // namespace portal_ml
