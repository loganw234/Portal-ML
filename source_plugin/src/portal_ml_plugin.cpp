#include "portal_ml_plugin.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
#include "engine/IEngineTrace.h"
#include "engine/iserverplugin.h"
#include "game/server/iplayerinfo.h"
#endif

namespace portal_ml {

namespace {
int ExtractIntField(const std::string& json, const std::string& key, int fallback) {
    const std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    return std::atoi(json.c_str() + pos);
}

std::string ExtractStringField(
    const std::string& json, const std::string& key, const std::string& fallback) {
    const std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    auto end = json.find('"', pos);
    if (end == std::string::npos) return fallback;
    return json.substr(pos, end - pos);
}
}  // namespace

void PortalMLPlugin::SendServerCmd(const std::string& cmd, bool execute_now) const {
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (engine) {
        engine->ServerCommand(cmd.c_str());
        if (execute_now) {
            engine->ServerExecute();
        }
    }
#else
    (void)cmd;
    (void)execute_now;
#endif
}

bool PortalMLPlugin::Load(
    void* engine_server, void* player_info_mgr, void* server_game_dll, void* engine_trace) {
    engine_ = engine_server;
    extractor_.SetEngineInterfaces(engine_server, player_info_mgr, server_game_dll, engine_trace);
    motor_.SetEngineServer(engine_server);
    return ipc_.StartTcp("127.0.0.1", 27500);
}

void PortalMLPlugin::Unload() {
    motor_.StopMovement();
    ipc_.Stop();
}

void PortalMLPlugin::OnLevelInit(const char* map_name) {
    current_map_name_ = map_name ? map_name : "";
    if (current_map_name_ == "portal_ml_5chambers") {
        bsp_map_ = BspMapData{};
        extractor_.SetGenericMap(nullptr);
        return;
    }
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (engine && !current_map_name_.empty()) {
        char gdir[512] = {};
        engine->GetGameDir(gdir, sizeof(gdir));
        const std::string bsp_path = std::string(gdir) + "/maps/" + current_map_name_ + ".bsp";
        if (BspScanner::LoadFromFile(bsp_path, current_map_name_, bsp_map_)) {
            extractor_.SetGenericMap(&bsp_map_);
            std::ostringstream log;
            log << "[Portal-ML] Generic BSP scan loaded '" << current_map_name_
                << "': entities=" << bsp_map_.entities.size()
                << ", portal_surfaces=" << bsp_map_.portal_surfaces.size() << "\n";
            engine->LogPrint(log.str().c_str());
        } else {
            extractor_.SetGenericMap(nullptr);
        }
    }
#endif
}

void PortalMLPlugin::ServerActivate() {
    SendServerCmd("sv_cheats 1; sv_gravity 600\n", false);
    if (!extractor_.IsGenericMode()) {
        ResetChamberInPlace(1);
    } else {
        achieved_milestones_.clear();
        extractor_.OnResetChamber(0);
    }
}

void PortalMLPlugin::ResetChamberInPlace(int chamber_id) {
    current_chamber_ = chamber_id;
    tick_ = 0;
    pending_subticks_ = 0;
    awaiting_post_step_ = false;
    awaiting_post_reset_ = false;
    pending_drop_cube_.clear();
    pending_drop_multi_btn_ = false;
    multi_btn_drops_done_ = 0;
    active_cmd_ = MotorCommand{};
    achieved_milestones_.clear();
    extractor_.OnResetChamber(chamber_id);
    extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);

    if (extractor_.IsGenericMode()) {
        if (chamber_id >= 2 &&
            static_cast<size_t>(chamber_id) <= bsp_map_.room_spawns.size()) {
            const auto& sp = bsp_map_.room_spawns[static_cast<size_t>(chamber_id - 1)];
            std::ostringstream ss;
            ss << "ent_fire !player AddOutput \"basevelocity 0 0 0\"; "
               << "give weapon_portalgun; "
               << "setpos " << sp.pos.x << ' ' << sp.pos.y << ' ' << (sp.pos.z + 4.0f)
               << "; setang 0 " << sp.yaw << " 0\n";
            SendServerCmd(ss.str(), false);
        }
        return;
    }

    const auto& layout = GetChamberLayout(chamber_id);
    std::ostringstream ss;
    // Sub-millisecond in-place reset (no changelevel):
    // 1. Stop movement, restore health (in case a prior episode fell into a hazard pit), and fizzle portals
    ss << "-forward; -use; ent_fire !player SetHealth 100; ent_fire !player AddOutput \"basevelocity 0 0 0\"; "
       << "ent_fire prop_portal Fizzle; "
       << "ent_fire portal_ml_blue SetActivatedState 0; "
       << "ent_fire portal_ml_orange SetActivatedState 0; ";
    // 2. Teleport player to chamber spawn
    ss << "setpos " << layout.player_spawn.x << ' ' << layout.player_spawn.y << ' '
       << layout.player_spawn.z << "; setang 0 " << layout.player_yaw << " 0; ";
    // 3. Reset chamber cube back to spawn via dedicated Teleport-Home point_teleport
    if (layout.has_cube) {
        const std::string tp = layout.cube_id + "_reset_tp";
        ss << "ent_fire " << tp << " Teleport; "
           << "ent_fire " << layout.cube_id << " AddOutput \"solid 6\" 0.02; "
           << "ent_fire " << layout.cube_id << " EnableMotion \"\" 0.03; "
           << "ent_fire " << layout.cube_id << " Wake \"\" 0.04; ";
    }
    // 4. Reset exit door to initial state
    ss << "ent_fire " << layout.exit_id << (layout.exit_starts_open ? " Open" : " Close") << "\n";
    SendServerCmd(ss.str(), false);
}

double PortalMLPlugin::ComputeStepReward(
    const BlackboardState& post_state, std::vector<std::string>& out_milestones) {
    double reward = -0.05;
    auto award = [&](const char* name, double pts) {
        if (achieved_milestones_.insert(name).second) {
            out_milestones.emplace_back(name);
            reward += pts;
        }
    };
    if (!post_state.held_entity_id.empty()) {
        award("cube_grabbed", 10.0);
    }
    for (const auto& e : post_state.entities) {
        if ((e.kind == "floor_button" || e.kind == "pedestal_button") && e.active) {
            award("button_pressed", 25.0);
        }
    }
    if (post_state.blue_portal.active) {
        award("portal_blue_placed", 5.0);
    }
    if (post_state.orange_portal.active) {
        award("portal_orange_placed", 5.0);
    }
    if (post_state.blue_portal.active && post_state.orange_portal.active &&
        post_state.blue_portal.zone_id != post_state.orange_portal.zone_id) {
        award("portals_bridged", 15.0);
    }
    if (post_state.player_zone_id != step_start_zone_) {
        award("portal_traversed", 20.0);
    }
    if (post_state.success) {
        award("exit_reached", 100.0);
    }
    return reward;
}

void PortalMLPlugin::GameFrame(bool simulating) {
    if (!simulating) return;

    ipc_.PollAccept();
    if (!ipc_.HasClient()) return;

    const float step_dt = tick_dt_ * static_cast<float>(subticks_per_step_);

    // Allow Source physics to execute queued commands across sub-ticks
    if (pending_subticks_ > 0) {
        --pending_subticks_;
        if (awaiting_post_reset_ && pending_subticks_ == 1 && !extractor_.IsGenericMode()) {
            // Re-pin player at spawn after velocity damping so zero residual momentum carries over
            const auto& layout = GetChamberLayout(current_chamber_);
            extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);
            std::ostringstream ss;
            ss << "ent_fire !player AddOutput \"basevelocity 0 0 0\"; "
               << "setpos " << layout.player_spawn.x << ' ' << layout.player_spawn.y << ' '
               << layout.player_spawn.z << "; setang 0 " << layout.player_yaw << " 0\n";
            SendServerCmd(ss.str(), false);
        }
        if (awaiting_post_step_ && active_cmd_.kind == MotorCmdKind::kDrop &&
            !pending_drop_cube_.empty()) {
            // Map synthetic multi-cube IDs ("box_2_0" -> "box_2") back to the real BSP targetname
            std::string wire_cube = pending_drop_cube_;
            if (extractor_.IsGenericMode() && pending_drop_multi_btn_) {
                const auto us = wire_cube.rfind('_');
                if (us != std::string::npos && us + 1 < wire_cube.size()) {
                    bool all_digits = true;
                    for (size_t k = us + 1; k < wire_cube.size(); ++k) {
                        if (wire_cube[k] < '0' || wire_cube[k] > '9') {
                            all_digits = false;
                            break;
                        }
                    }
                    if (all_digits) {
                        wire_cube = wire_cube.substr(0, us);
                    }
                }
            }
            if (extractor_.IsGenericMode() && pending_subticks_ == 24 &&
                pending_drop_multi_btn_ && multi_btn_drops_done_ > 0) {
                // Previous button's cube was renamed to portal_ml_pinned_*; spawn a fresh
                // matching cube with the map's required filter targetname so dropping on
                // button #2 never yanks button #1's cube off its pad.
                std::ostringstream ss;
                ss << "ent_create prop_physics model \"models/props/metal_box.mdl\" targetname \""
                   << wire_cube << "\"\n";
                SendServerCmd(ss.str(), false);
            } else if (extractor_.IsGenericMode() && pending_subticks_ == 20) {
                // Step 1 (after ~0.51s off button so Valve's player-trigger finishes
                // OnEndTouchAll + floor_button_door_* finishes rising): configure
                // point_teleport 28u above tgt_z and wake VPhysics while STILL solid=0.
                std::ostringstream ss;
                ss << "ent_fire point_teleport AddOutput \"angles 0 0 0\"; "
                   << "ent_fire point_teleport AddOutput \"origin "
                   << pending_drop_target_.x << ' ' << pending_drop_target_.y << ' '
                   << (pending_drop_target_.z + 28.0f) << "\"; "
                   << "ent_fire point_teleport AddOutput \"target " << wire_cube << "\"; "
                   << "ent_fire " << wire_cube << " EnableMotion\n";
                SendServerCmd(ss.str(), false);
            } else if (extractor_.IsGenericMode() && pending_subticks_ == 18) {
                // Step 2: Teleport while STILL solid=0 to dead-center above button pad.
                SendServerCmd("ent_fire point_teleport Teleport\n", false);
            } else if (extractor_.IsGenericMode() && pending_subticks_ == 17) {
                // Step 3: freeze dead-center above the pad to zero linear/angular velocity.
                SendServerCmd("ent_fire " + wire_cube + " DisableMotion\n", false);
            } else if (extractor_.IsGenericMode() && pending_subticks_ == 16) {
                // Step 4: restore SOLID_VPHYSICS on a distinct tick while DisableMotion is active.
                SendServerCmd("ent_fire " + wire_cube + " AddOutput \"solid 6\"\n", false);
            } else if (extractor_.IsGenericMode() && pending_subticks_ == 15) {
                // Step 5: re-enable VPhysics + Wake so Valve's floor_button_trigger_box_*
                // registers OnStartTouch and opens floor_button_door_* / exit door.
                std::ostringstream ss;
                ss << "ent_fire " << wire_cube << " EnableMotion; "
                   << "ent_fire " << wire_cube << " Wake \"\" 0.01; "
                   << "ent_fire point_teleport Kill \"\" 0.04\n";
                SendServerCmd(ss.str(), false);
            } else if (extractor_.IsGenericMode() && pending_subticks_ == 12) {
                // Step 6: pin cube in place on the button pad after 3 physics ticks (~0.045s)
                // once Valve's trigger has fired, preventing duplicate dropper cubes sharing
                // targetname ("box") from blasting each other off the button rim.
                std::ostringstream ss;
                ss << "ent_fire " << wire_cube << " DisableMotion";
                if (pending_drop_multi_btn_) {
                    ss << "; ent_fire " << wire_cube << " AddOutput \"targetname portal_ml_pinned_"
                       << multi_btn_drops_done_ << "\" 0.02";
                    ++multi_btn_drops_done_;
                }
                ss << "\n";
                SendServerCmd(ss.str(), false);
                extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);
                extractor_.NotifyHoldState("");
                pending_drop_cube_.clear();
            } else if (!extractor_.IsGenericMode() && pending_subticks_ == subticks_per_step_ - 2) {
                // Player was already stepped 58u clear and aimed at the drop target on tick 0;
                // now ent_teleport places the solid cube flush on top of the floor button with
                // zero player-hull overlap.
                std::ostringstream ss;
                ss << "ent_teleport " << pending_drop_cube_ << "; "
                   << "ent_fire " << pending_drop_cube_ << " DisableMotion \"\" 0.01\n";
                SendServerCmd(ss.str(), false);
                extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);
                extractor_.NotifyHoldState("");
                pending_drop_cube_.clear();
            }
        }
        if (awaiting_post_step_ &&
            (active_cmd_.kind == MotorCmdKind::kMoveTo || active_cmd_.kind == MotorCmdKind::kMoveVec) &&
            pending_subticks_ > 0) {
            const auto mid_state = extractor_.Extract(current_chamber_, tick_, tick_dt_);
            if (mid_state.player_zone_id != step_start_zone_) {
                // Player just traversed a portal — finish step immediately without oscillating
                extractor_.SetPlayerVelocityXY(0.0f, 0.0f);
                pending_subticks_ = 0;
            }
        }
        if (pending_subticks_ > 0) {
            return;
        }
        if (awaiting_post_reset_) {
            extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);
            awaiting_post_reset_ = false;
            const auto state = extractor_.Extract(current_chamber_, tick_, step_dt);
            ipc_.SendFrame("{\"ok\":true,\"op\":\"reset\",\"snapshot\":" +
                           BlackboardExtractor::ToJson(state) + "}");
            return;
        }
        if (awaiting_post_step_) {
            awaiting_post_step_ = false;
            const auto post_state = extractor_.Extract(current_chamber_, tick_, step_dt);
            // Ensure the physical Source func_door slides open when the cube rests on the floor button
            if (!extractor_.IsGenericMode()) {
                const auto& layout = GetChamberLayout(current_chamber_);
                if (layout.has_floor_button) {
                    for (const auto& e : post_state.entities) {
                        if (e.kind == "floor_button" && e.active) {
                            SendServerCmd("ent_fire " + layout.exit_id + " Open\n", false);
                            break;
                        }
                    }
                }
            }
            std::vector<std::string> step_milestones;
            const double reward = ComputeStepReward(post_state, step_milestones);
            std::ostringstream resp;
            resp << "{\"ok\":true,\"op\":\"step\",\"snapshot\":"
                 << BlackboardExtractor::ToJson(post_state)
                 << ",\"reward\":" << reward
                 << ",\"done\":" << (post_state.done ? "true" : "false")
                 << ",\"success\":" << (post_state.success ? "true" : "false")
                 << ",\"milestones\":[";
            for (size_t i = 0; i < step_milestones.size(); ++i) {
                if (i > 0) resp << ',';
                resp << '"' << step_milestones[i] << '"';
            }
            resp << "]}";
            ipc_.SendFrame(resp.str());
            return;
        }
    }

    std::string req;
    if (!ipc_.RecvFrame(req)) {
        return;
    }

    if (req.find("\"op\":\"ping\"") != std::string::npos) {
        ipc_.SendFrame("{\"ok\":true,\"op\":\"pong\"}");
        return;
    }

    if (req.find("\"op\":\"map_info\"") != std::string::npos) {
        std::ostringstream info;
        info << "{\"ok\":true,\"op\":\"map_info\",\"map_name\":\"" << current_map_name_
             << "\",\"generic_mode\":" << (extractor_.IsGenericMode() ? "true" : "false")
             << ",\"can_fire_blue\":" << (bsp_map_.can_fire_blue ? "true" : "false")
             << ",\"can_fire_orange\":" << (bsp_map_.can_fire_orange ? "true" : "false")
             << ",\"discovered_entities\":" << bsp_map_.entities.size()
             << ",\"portal_surfaces\":" << bsp_map_.portal_surfaces.size() << '}';
        ipc_.SendFrame(info.str());
        return;
    }

    if (req.find("\"op\":\"reset\"") != std::string::npos) {
        const int cid = ExtractIntField(req, "chamber_id", 1);
        ResetChamberInPlace(cid);
        // Wait 6 ticks so setpos + velocity zero + cube/door reset settle cleanly
        pending_subticks_ = 6;
        awaiting_post_reset_ = true;
        return;
    }

    if (req.find("\"op\":\"step\"") != std::string::npos) {
        ++tick_;
        const auto pre_state = extractor_.Extract(current_chamber_, tick_, step_dt);
        step_start_zone_ = pre_state.player_zone_id;
        active_cmd_ = MotorController::ParseCommandJson(req);
        if (active_cmd_.kind == MotorCmdKind::kDrop) {
            pending_drop_cube_ = pre_state.held_entity_id;
            int fb_count = 0;
            for (const auto& e : pre_state.entities) {
                if (e.kind == "floor_button") ++fb_count;
            }
            pending_drop_multi_btn_ = (fb_count >= 2);
            pending_drop_target_ = {
                pre_state.player_pos.x + std::cos(pre_state.player_yaw * (3.14159265f / 180.0f)) * 48.0f,
                pre_state.player_pos.y + std::sin(pre_state.player_yaw * (3.14159265f / 180.0f)) * 48.0f,
                pre_state.player_pos.z + 12.0f,
            };
            for (const auto& e : pre_state.entities) {
                if (e.kind == "floor_button" && e.zone_id == pre_state.player_zone_id) {
                    const float dx = e.pos.x - pre_state.player_pos.x;
                    const float dy = e.pos.y - pre_state.player_pos.y;
                    if (std::hypot(dx, dy) <= 56.0f) {
                        pending_drop_target_ = {e.pos.x, e.pos.y, e.pos.z + 12.0f};
                        break;
                    }
                }
            }
        }
        motor_.Execute(active_cmd_, pre_state, extractor_);
        if (active_cmd_.kind == MotorCmdKind::kDrop && extractor_.IsGenericMode()) {
            pending_subticks_ = 54;
        } else {
            pending_subticks_ = subticks_per_step_;
        }
        awaiting_post_step_ = true;
        return;
    }

    if (req.find("\"op\":\"start_movie\"") != std::string::npos) {
        const std::string prefix = ExtractStringField(req, "prefix", "pml_rec");
        const int fps = ExtractIntField(req, "fps", 60);
        std::string safe_prefix;
        for (char c : prefix) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '-') {
                safe_prefix.push_back(c);
            }
        }
        if (safe_prefix.empty()) safe_prefix = "pml_rec";
        std::ostringstream cmd;
        cmd << "m_pitch 0; m_yaw 0; setang 0 0 0; host_framerate "
            << (fps > 0 ? fps : 60) << "; startmovie " << safe_prefix << "\n";
        SendServerCmd(cmd.str(), false);
        std::string gdir_str;
#ifdef PORTAL_ML_USE_SOURCE_SDK
        if (auto* engine = static_cast<IVEngineServer*>(engine_)) {
            char gdir[512] = {};
            engine->GetGameDir(gdir, sizeof(gdir));
            gdir_str = gdir;
            for (char& ch : gdir_str) {
                if (ch == '\\') ch = '/';
            }
        }
#endif
        ipc_.SendFrame("{\"ok\":true,\"op\":\"start_movie\",\"prefix\":\"" + safe_prefix +
                       "\",\"game_dir\":\"" + gdir_str + "\"}");
        return;
    }

    if (req.find("\"op\":\"end_movie\"") != std::string::npos) {
        SendServerCmd("endmovie; host_framerate 0; m_pitch 0.022; m_yaw 0.022\n", false);
        ipc_.SendFrame("{\"ok\":true,\"op\":\"end_movie\"}");
        return;
    }

    if (req.find("\"op\":\"shutdown\"") != std::string::npos) {
        ipc_.SendFrame("{\"ok\":true,\"op\":\"shutdown\"}");
        ipc_.DisconnectClient();
        return;
    }

    ipc_.SendFrame("{\"ok\":false,\"error\":\"Unsupported operation\"}");
}

}  // namespace portal_ml

#ifdef PORTAL_ML_USE_SOURCE_SDK

// Official Valve IServerPluginCallbacks (ISERVERPLUGINCALLBACKS003) adapter.
// Exported directly via CreateInterface with zero tier0/tier1 .lib dependencies.
class CPortalMLServerPlugin final : public IServerPluginCallbacks {
public:
    bool Load(CreateInterfaceFn interfaceFactory, CreateInterfaceFn gameServerFactory) override {
        void* engine = interfaceFactory ? interfaceFactory(INTERFACEVERSION_VENGINESERVER_VERSION_21, nullptr) : nullptr;
        void* trace = interfaceFactory ? interfaceFactory(INTERFACEVERSION_ENGINETRACE_SERVER, nullptr) : nullptr;
        void* pim = gameServerFactory ? gameServerFactory(INTERFACEVERSION_PLAYERINFOMANAGER, nullptr) : nullptr;
        void* gamedll = gameServerFactory ? gameServerFactory(INTERFACEVERSION_SERVERGAMEDLL_VERSION_9, nullptr) : nullptr;
        const bool ok = impl_.Load(engine, pim, gamedll, trace);
        if (engine) {
            static_cast<IVEngineServer*>(engine)->LogPrint(
                ok ? "[Portal-ML] Plugin loaded; IPC listening on 127.0.0.1:27500\n"
                   : "[Portal-ML] ERROR: Failed to bind IPC on 127.0.0.1:27500\n");
        }
        return ok;
    }

    void Unload() override { impl_.Unload(); }
    void Pause() override {}
    void UnPause() override {}
    const char* GetPluginDescription() override {
        return "Portal-ML Decoupled Neuroevolutionary Agent Bridge v0.1";
    }
    void LevelInit(char const* pMapName) override { impl_.OnLevelInit(pMapName); }
    void ServerActivate(edict_t* /*pEdictList*/, int /*edictCount*/, int /*clientMax*/) override {
        impl_.ServerActivate();
    }
    void GameFrame(bool simulating) override { impl_.GameFrame(simulating); }
    void LevelShutdown() override {}
    void ClientActive(edict_t* /*pEntity*/) override {}
    void ClientDisconnect(edict_t* /*pEntity*/) override {}
    void ClientPutInServer(edict_t* /*pEntity*/, char const* /*playername*/) override {}
    void SetCommandClient(int /*index*/) override {}
    void ClientSettingsChanged(edict_t* /*pEdict*/) override {}
    PLUGIN_RESULT ClientConnect(bool* bAllowConnect, edict_t* /*pEntity*/, const char* /*pszName*/,
                                const char* /*pszAddress*/, char* /*reject*/, int /*maxrejectlen*/) override {
        if (bAllowConnect) *bAllowConnect = true;
        return PLUGIN_CONTINUE;
    }
    PLUGIN_RESULT ClientCommand(edict_t* /*pEntity*/, const CCommand& /*args*/) override {
        return PLUGIN_CONTINUE;
    }
    PLUGIN_RESULT NetworkIDValidated(const char* /*pszUserName*/, const char* /*pszNetworkID*/) override {
        return PLUGIN_CONTINUE;
    }
    void OnQueryCvarValueFinished(QueryCvarCookie_t /*iCookie*/, edict_t* /*pPlayerEntity*/,
                                  EQueryCvarValueStatus /*eStatus*/, const char* /*pCvarName*/,
                                  const char* /*pCvarValue*/) override {}
    void OnEdictAllocated(edict_t* /*edict*/) override {}
    void OnEdictFreed(const edict_t* /*edict*/) override {}

private:
    portal_ml::PortalMLPlugin impl_;
};

static CPortalMLServerPlugin g_PortalMLPlugin;

extern "C" __declspec(dllexport) void* CreateInterface(const char* pName, int* pReturnCode) {
    if (pName &&
        (std::strcmp(pName, INTERFACEVERSION_ISERVERPLUGINCALLBACKS) == 0 ||
         std::strcmp(pName, INTERFACEVERSION_ISERVERPLUGINCALLBACKS_VERSION_2) == 0 ||
         std::strcmp(pName, INTERFACEVERSION_ISERVERPLUGINCALLBACKS_VERSION_1) == 0)) {
        if (pReturnCode) *pReturnCode = IFACE_OK;
        return static_cast<IServerPluginCallbacks*>(&g_PortalMLPlugin);
    }
    if (pReturnCode) *pReturnCode = IFACE_FAILED;
    return nullptr;
}

#endif
