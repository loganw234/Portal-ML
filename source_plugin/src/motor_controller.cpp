#include "motor_controller.h"
#include "bsp_scanner.h"

#include <cmath>
#include <cstdlib>
#include <sstream>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
#endif

namespace portal_ml {

namespace {
constexpr float kRadToDeg = 57.2957795f;
constexpr float kMoveSpeed = 240.0f;

float ClampF(float v, float lo, float hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

float WrapDeg(float deg) {
    float r = std::fmod(deg + 180.0f, 360.0f);
    if (r < 0.0f) r += 360.0f;
    return r - 180.0f;
}

float ExtractJsonFloatField(const std::string& json, const std::string& key, float fallback = 0.0f) {
    const std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    return static_cast<float>(std::atof(json.c_str() + pos));
}

std::string ExtractJsonStringField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    auto end = json.find('"', pos);
    if (end == std::string::npos) return {};
    return json.substr(pos, end - pos);
}

bool FindPortalApproach(const BlackboardState& state, const PortalRecord& portal, Vec3& out_pos) {
    if (!portal.active) return false;
    const float rx = state.player_pos.x - portal.pos.x;
    const float ry = state.player_pos.y - portal.pos.y;
    // Distance along wall normal and laterally parallel to wall
    const float along_n = rx * portal.normal.x + ry * portal.normal.y;
    const float lateral = std::fabs(-rx * portal.normal.y + ry * portal.normal.x);
    if (lateral > 28.0f && along_n < 44.0f) {
        // Line up cleanly in front of portal before walking straight through
        out_pos = {
            portal.pos.x + portal.normal.x * 56.0f,
            portal.pos.y + portal.normal.y * 56.0f,
            portal.pos.z,
        };
    } else {
        out_pos = {portal.pos.x, portal.pos.y, portal.pos.z};
    }
    return true;
}

bool FindTargetPos(const BlackboardState& state, const std::string& id, Vec3& out_pos) {
    if (id == "portal:blue") {
        return FindPortalApproach(state, state.blue_portal, out_pos);
    }
    if (id == "portal:orange") {
        return FindPortalApproach(state, state.orange_portal, out_pos);
    }
    for (const auto& e : state.entities) {
        if (e.id == id) {
            out_pos = e.pos;
            return true;
        }
    }
    return false;
}
}  // namespace

void MotorController::IssueCmd(const std::string& cmd) const {
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (engine) {
        engine->ServerCommand(cmd.c_str());
    }
#else
    (void)cmd;
#endif
}

void MotorController::StopMovement() const {
    IssueCmd("-forward; -use; ent_fire !player AddOutput \"basevelocity 0 0 0\"\n");
}

MotorCommand MotorController::ParseCommandJson(const std::string& json) {
    MotorCommand cmd;
    const std::string kind = ExtractJsonStringField(json, "cmd");
    if (kind == "move_to") {
        cmd.kind = MotorCmdKind::kMoveTo;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
    } else if (kind == "move_vec") {
        cmd.kind = MotorCmdKind::kMoveVec;
        cmd.forward = ExtractJsonFloatField(json, "forward", 0.0f);
        cmd.strafe = ExtractJsonFloatField(json, "strafe", 0.0f);
        cmd.yaw_delta = ExtractJsonFloatField(json, "yaw_delta", 0.0f);
    } else if (kind == "grab") {
        cmd.kind = MotorCmdKind::kGrab;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
    } else if (kind == "drop") {
        cmd.kind = MotorCmdKind::kDrop;
    } else if (kind == "press") {
        cmd.kind = MotorCmdKind::kPress;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
    } else if (kind == "shoot_portal") {
        cmd.kind = MotorCmdKind::kShootPortal;
        cmd.color = ExtractJsonStringField(json, "color");
        cmd.wall_id = ExtractJsonStringField(json, "wall_id");
    }
    return cmd;
}

void MotorController::Execute(
    const MotorCommand& cmd,
    const BlackboardState& state,
    BlackboardExtractor& extractor) const {
    if (cmd.kind == MotorCmdKind::kNoOp) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        StopMovement();
        return;
    }

    if (cmd.kind == MotorCmdKind::kMoveVec) {
        // Phase 2 continuous body-frame control from neuroevolved MLP
        float fwd = ClampF(cmd.forward, -1.0f, 1.0f);
        float stf = ClampF(cmd.strafe, -1.0f, 1.0f);
        const float yd = ClampF(cmd.yaw_delta, -1.0f, 1.0f);
        const float mag = std::hypot(fwd, stf);
        if (mag > 1.0f) {
            fwd /= mag;
            stf /= mag;
        }
        const float new_yaw = WrapDeg(state.player_yaw + yd * 36.0f);
        extractor.SetPlayerEyeAngles(0.0f, new_yaw);

        const float rad = new_yaw / kRadToDeg;
        const float cos_y = std::cos(rad);
        const float sin_y = std::sin(rad);
        const float target_vx = (fwd * cos_y - stf * sin_y) * kMoveSpeed;
        const float target_vy = (fwd * sin_y + stf * cos_y) * kMoveSpeed;
        const float dvx = ClampF(target_vx - state.player_vel.x, -260.0f, 260.0f);
        const float dvy = ClampF(target_vy - state.player_vel.y, -260.0f, 260.0f);

        // Native SendProp m_vecBaseVelocity + FL_BASEVELOCITY actuation (zero ent_fire velocity spam)
        extractor.ApplyBaseVelocityXY(dvx, dvy);
        if (!state.held_entity_id.empty()) {
            const float cx = state.player_pos.x;
            const float cy = state.player_pos.y;
            const float cz = state.player_pos.z + 108.0f;
            std::ostringstream ss;
            ss << "ent_fire " << state.held_entity_id << " AddOutput \"origin "
               << cx << ' ' << cy << ' ' << cz << "\"\n";
            IssueCmd(ss.str());
        }
        return;
    }

    if (cmd.kind == MotorCmdKind::kMoveTo) {
        Vec3 target{};
        if (!FindTargetPos(state, cmd.entity_id, target)) {
            extractor.SetPlayerVelocityXY(0.0f, 0.0f);
            StopMovement();
            return;
        }
        const float dx = target.x - state.player_pos.x;
        const float dy = target.y - state.player_pos.y;
        const float dist = std::hypot(dx, dy);
        if (dist < 1e-3f) {
            extractor.SetPlayerVelocityXY(0.0f, 0.0f);
            StopMovement();
            return;
        }
        float nx = dx / dist;
        float ny = dy / dist;
        const bool is_portal = (cmd.entity_id == "portal:blue" || cmd.entity_id == "portal:orange");
        // Generic pit-edge avoidance: if walking straight toward a zone-0 target drops
        // into a sunken pit (zone 1), steer wide around the pit edge upon returning to zone 0.
        if (extractor.IsGenericMode()) {
            if (!is_portal && prev_zone_ == 0 && state.player_zone_id == 1 && !last_target_id_.empty()) {
                if (pit_avoid_target_ == last_target_id_ && pit_avoid_steps_ > 0) {
                    pit_avoid_sign_ = -pit_avoid_sign_;
                } else {
                    pit_avoid_sign_ = 1.0f;
                }
                pit_avoid_target_ = last_target_id_;
                pit_avoid_steps_ = 16;
            }
            prev_zone_ = state.player_zone_id;
        }
        // Generic Bug-style obstacle wall-following on arbitrary campaign maps when
        // forward progress toward target stalls against glass partitions / corners / doorframes.
        if (extractor.IsGenericMode() && !is_portal && dist > 48.0f && state.tick > 2) {
            if (cmd.entity_id != last_target_id_) {
                last_target_id_ = cmd.entity_id;
                best_target_dist_ = dist;
                stall_steps_ = 0;
                slide_remaining_ = 0;
            }
            if (dist < best_target_dist_ - 12.0f) {
                best_target_dist_ = dist;
                stall_steps_ = 0;
                slide_remaining_ = 0;
            } else {
                const float cur_spd = std::hypot(state.player_vel.x, state.player_vel.y);
                const float toward_spd = state.player_vel.x * nx + state.player_vel.y * ny;
                if (toward_spd < 40.0f || cur_spd < 30.0f) {
                    ++stall_steps_;
                    if (stall_steps_ >= 2 && stall_steps_ < 12) {
                        slide_remaining_ = 5;
                    } else if (stall_steps_ == 12) {
                        // Stage 2: trapped in a U-shaped pocket — back out 135 deg for 5 steps
                        // then resume sliding around the outside of the pocket.
                        slide_remaining_ = 10;
                    } else if (stall_steps_ >= 26) {
                        slide_sign_ = -slide_sign_;
                        stall_steps_ = 2;
                    }
                }
            }
            if (slide_remaining_ > 0) {
                --slide_remaining_;
                const float offset = (stall_steps_ >= 12 && slide_remaining_ >= 5) ? 2.35f : 1.40f;
                const float angle = std::atan2(ny, nx) + slide_sign_ * offset;
                nx = std::cos(angle);
                ny = std::sin(angle);
            } else if (state.player_zone_id == 0 && pit_avoid_steps_ > 0 &&
                       cmd.entity_id == pit_avoid_target_ && dist > 160.0f) {
                --pit_avoid_steps_;
                const float angle = std::atan2(ny, nx) + pit_avoid_sign_ * 0.55f;
                nx = std::cos(angle);
                ny = std::sin(angle);
            }
        } else {
            last_target_id_.clear();
            best_target_dist_ = 1e9f;
            stall_steps_ = 0;
            slide_remaining_ = 0;
        }
        const float speed = (!is_portal && dist < 72.0f) ? 140.0f : kMoveSpeed;
        const float target_vx = nx * speed;
        const float target_vy = ny * speed;
        const float dvx = ClampF(target_vx - state.player_vel.x, -260.0f, 260.0f);
        const float dvy = ClampF(target_vy - state.player_vel.y, -260.0f, 260.0f);
        const float yaw = std::atan2(dy, dx) * kRadToDeg;
        extractor.SetPlayerEyeAngles(0.0f, yaw);

        extractor.ApplyBaseVelocityXY(dvx, dvy);
        if (!state.held_entity_id.empty()) {
            const float cx = state.player_pos.x;
            const float cy = state.player_pos.y;
            const float cz = state.player_pos.z + 108.0f;
            std::ostringstream ss;
            ss << "ent_fire " << state.held_entity_id << " AddOutput \"origin "
               << cx << ' ' << cy << ' ' << cz << "\"\n";
            IssueCmd(ss.str());
        }
        return;
    }

    if (cmd.kind == MotorCmdKind::kGrab) {
        Vec3 target{};
        if (!FindTargetPos(state, cmd.entity_id, target)) {
            extractor.SetPlayerVelocityXY(0.0f, 0.0f);
            StopMovement();
            return;
        }
        const float dx = target.x - state.player_pos.x;
        const float dy = target.y - state.player_pos.y;
        const float yaw = std::atan2(dy, dx) * kRadToDeg;
        const float cx = state.player_pos.x;
        const float cy = state.player_pos.y;
        const float cz = state.player_pos.z + 108.0f;

        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        std::ostringstream ss;
        ss << "setang 0 " << yaw << " 0; "
           << "ent_fire !player AddOutput \"basevelocity 0 0 0\"; "
           << "ent_fire " << cmd.entity_id << " DisableMotion; "
           << "ent_fire " << cmd.entity_id << " AddOutput \"solid 0\"; "
           << "ent_fire " << cmd.entity_id << " AddOutput \"origin "
           << cx << ' ' << cy << ' ' << cz << "\"\n";
        IssueCmd(ss.str());
        extractor.NotifyHoldState(cmd.entity_id);
        return;
    }

    if (cmd.kind == MotorCmdKind::kDrop) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f, true);
        std::ostringstream ss;
        ss << "ent_fire !player AddOutput \"basevelocity 0 0 0\"; ";
        if (!state.held_entity_id.empty()) {
            // Compute drop target (floor button if in range, else 48u ahead of player)
            // and step the player 56u clear FIRST so the solid cube never overlaps the
            // player's 16u hull when ent_teleport places it on the button.
            float tgt_x = state.player_pos.x + std::cos(state.player_yaw * (3.14159265f / 180.0f)) * 48.0f;
            float tgt_y = state.player_pos.y + std::sin(state.player_yaw * (3.14159265f / 180.0f)) * 48.0f;
            float tgt_z = 12.0f;
            for (const auto& e : state.entities) {
                if (e.kind == "floor_button" && e.zone_id == state.player_zone_id) {
                    const float dx = e.pos.x - state.player_pos.x;
                    const float dy = e.pos.y - state.player_pos.y;
                    if (std::hypot(dx, dy) <= 56.0f) {
                        tgt_x = e.pos.x;
                        tgt_y = e.pos.y;
                        tgt_z = e.pos.z + 12.0f;
                    }
                    break;
                }
            }
            float bx = -1.0f;
            float by = 0.0f;
            if (extractor.IsGenericMode()) {
                // Step back toward map spawn (guaranteed open room space, avoiding cube overlap)
                const auto* gmap = extractor.GetGenericMap();
                if (gmap) {
                    bx = gmap->player_spawn.x - tgt_x;
                    by = gmap->player_spawn.y - tgt_y;
                }
            } else {
                const auto& layout = GetChamberLayout(state.chamber_id);
                bx = layout.exit_pos.x - tgt_x;
                by = layout.exit_pos.y - tgt_y;
            }
            float bdist = std::hypot(bx, by);
            if (bdist < 1e-3f) {
                bx = -1.0f;
                by = 0.0f;
                bdist = 1.0f;
            }
            const float clear_dist = extractor.IsGenericMode() ? 60.0f : 72.0f;
            float clear_x = tgt_x + (bx / bdist) * clear_dist;
            float clear_y = tgt_y + (by / bdist) * clear_dist;
            float clear_z = extractor.IsGenericMode() ? (tgt_z + 4.0f) : 16.0f;
            if (!extractor.IsGenericMode()) {
                const auto& layout = GetChamberLayout(state.chamber_id);
                for (const auto& z : layout.zones) {
                    if (z.zone_id == state.player_zone_id) {
                        clear_x = ClampF(clear_x, z.min_x + 24.0f, z.max_x - 24.0f);
                        clear_y = ClampF(clear_y, z.min_y + 24.0f, z.max_y - 24.0f);
                        break;
                    }
                }
            }
            const float dx = tgt_x - clear_x;
            const float dy = tgt_y - clear_y;
            const float d = std::max(18.0f, std::hypot(dx, dy));
            const float drop_yaw = std::atan2(dy, dx) * kRadToDeg;
            const float drop_pitch = std::atan2((clear_z + 64.0f) - tgt_z, d) * kRadToDeg;

            ss << "setpos " << clear_x << ' ' << clear_y << ' ' << clear_z << "; ";
            extractor.SetPlayerEyeAngles(drop_pitch, drop_yaw);
            if (extractor.IsGenericMode()) {
                ss << "ent_create point_teleport";
            } else {
                ss << "ent_fire " << state.held_entity_id << " AddOutput \"solid 6\"";
            }
        }
        ss << "\n";
        IssueCmd(ss.str());
        return;
    }

    if (cmd.kind == MotorCmdKind::kPress) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        std::ostringstream ss;
        ss << "ent_fire " << cmd.entity_id << " Press\n";
        IssueCmd(ss.str());
        extractor.NotifyPedestalPressed(8.0f);
        return;
    }

    if (cmd.kind == MotorCmdKind::kShootPortal) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        for (const auto& w : state.portal_walls) {
            if (w.id == cmd.wall_id) {
                if (extractor.IsGenericMode()) {
                    // Unmodified Valve campaign maps: aim the player's real weapon_portalgun
                    // directly at the BSP concrete surface and fire FirePortal1 / FirePortal2.
                    const float dx = w.pos.x - state.player_pos.x;
                    const float dy = w.pos.y - state.player_pos.y;
                    const float dz = w.pos.z - (state.player_pos.z + 64.0f);
                    const float dxy = std::max(1.0f, std::hypot(dx, dy));
                    const float aim_yaw = std::atan2(dy, dx) * kRadToDeg;
                    const float aim_pitch = -std::atan2(dz, dxy) * kRadToDeg;
                    extractor.SetPlayerEyeAngles(aim_pitch, aim_yaw);
                    const char* fire_input = (cmd.color == "blue") ? "FirePortal1" : "FirePortal2";
                    std::ostringstream ss;
                    ss << "ent_fire weapon_portalgun " << fire_input << " \"\" 0.02\n";
                    IssueCmd(ss.str());
                    extractor.NotifyPortalPlacedWall(cmd.color, w);
                    return;
                }
                const float yaw = std::atan2(w.normal.y, w.normal.x) * kRadToDeg;
                // Place flush on the front face of the 1u concrete panel
                const float px = w.pos.x + w.normal.x * 1.5f;
                const float py = w.pos.y + w.normal.y * 1.5f;
                const float pz = w.pos.z;
                const std::string ent_name = (cmd.color == "blue") ? "portal_ml_blue" : "portal_ml_orange";
                std::ostringstream ss;
                ss << "ent_fire " << ent_name << " NewLocation \""
                   << px << ' ' << py << ' ' << pz << " 0 " << yaw << " 0\"; "
                   << "ent_fire " << ent_name << " SetActivatedState 1\n";
                IssueCmd(ss.str());
                extractor.NotifyPortalPlacedWall(cmd.color, w);
                return;
            }
        }
    }
}

}  // namespace portal_ml
