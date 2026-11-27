#include "portal_ml_plugin.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
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

bool PortalMLPlugin::Load(void* engine_server, void* player_info_mgr, void* server_game_dll) {
    engine_ = engine_server;
    extractor_.SetEngineInterfaces(engine_server, player_info_mgr, server_game_dll);
    motor_.SetEngineServer(engine_server);
    return ipc_.StartTcp("127.0.0.1", 27500);
}

void PortalMLPlugin::Unload() {
    motor_.StopMovement();
    ipc_.Stop();
}

void PortalMLPlugin::ServerActivate() {
    SendServerCmd("sv_cheats 1; sv_gravity 600\n", false);
    ResetChamberInPlace(1);
}

void PortalMLPlugin::ResetChamberInPlace(int chamber_id) {
    current_chamber_ = chamber_id;
    tick_ = 0;
    pending_subticks_ = 0;
    awaiting_post_step_ = false;
    awaiting_post_reset_ = false;
    pending_drop_cube_.clear();
    active_cmd_ = MotorCommand{};
    extractor_.OnResetChamber(chamber_id);
    extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);

    const auto& layout = GetChamberLayout(chamber_id);
    std::ostringstream ss;
    // Sub-millisecond in-place reset (no changelevel):
    // 1. Stop movement and fizzle portals
    ss << "-forward; -use; ent_fire !player AddOutput \"basevelocity 0 0 0\"; "
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

void PortalMLPlugin::GameFrame(bool simulating) {
    if (!simulating) return;

    ipc_.PollAccept();
    if (!ipc_.HasClient()) return;

    const float step_dt = tick_dt_ * static_cast<float>(subticks_per_step_);

    // Allow Source physics to execute queued commands across sub-ticks
    if (pending_subticks_ > 0) {
        --pending_subticks_;
        if (awaiting_post_reset_ && pending_subticks_ == 1) {
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
            !pending_drop_cube_.empty() && pending_subticks_ == subticks_per_step_ - 2) {
            // Teleport cube onto the floor/button, step player clear, and lift the cube's center
            // +20.5u above the 12u pad so it rests cleanly on top without CC_Ent_Teleport floor pop.
            const auto drop_state = extractor_.Extract(current_chamber_, tick_, tick_dt_);
            const auto& layout = GetChamberLayout(current_chamber_);
            float bx = layout.exit_pos.x - drop_state.player_pos.x;
            float by = layout.exit_pos.y - drop_state.player_pos.y;
            float bdist = std::hypot(bx, by);
            if (bdist < 1e-3f) {
                bx = 1.0f;
                by = 0.0f;
                bdist = 1.0f;
            }
            const float clear_x = drop_state.player_pos.x + (bx / bdist) * 52.0f;
            const float clear_y = drop_state.player_pos.y + (by / bdist) * 52.0f;
            std::ostringstream ss;
            ss << "ent_teleport " << pending_drop_cube_ << "; "
               << "setpos " << clear_x << ' ' << clear_y << " 16; "
               << "ent_fire " << pending_drop_cube_ << " DisableMotion \"\" 0.01\n";
            SendServerCmd(ss.str(), false);
            extractor_.SetPlayerVelocityXY(0.0f, 0.0f, true);
            extractor_.NotifyHoldState("");
            pending_drop_cube_.clear();
        }
        if (awaiting_post_step_ && active_cmd_.kind == MotorCmdKind::kMoveTo && pending_subticks_ > 0) {
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
            const auto& layout = GetChamberLayout(current_chamber_);
            if (layout.has_floor_button) {
                for (const auto& e : post_state.entities) {
                    if (e.kind == "floor_button" && e.active) {
                        SendServerCmd("ent_fire " + layout.exit_id + " Open\n", false);
                        break;
                    }
                }
            }
            ipc_.SendFrame(
                "{\"ok\":true,\"op\":\"step\",\"snapshot\":" +
                BlackboardExtractor::ToJson(post_state) +
                ",\"reward\":" + (post_state.success ? "99.95" : "-0.05") +
                ",\"done\":" + (post_state.done ? "true" : "false") +
                ",\"success\":" + (post_state.success ? "true" : "false") +
                ",\"milestones\":[]}");
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
        }
        motor_.Execute(active_cmd_, pre_state, extractor_);
        pending_subticks_ = subticks_per_step_;
        awaiting_post_step_ = true;
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
        void* pim = gameServerFactory ? gameServerFactory(INTERFACEVERSION_PLAYERINFOMANAGER, nullptr) : nullptr;
        void* gamedll = gameServerFactory ? gameServerFactory(INTERFACEVERSION_SERVERGAMEDLL_VERSION_9, nullptr) : nullptr;
        const bool ok = impl_.Load(engine, pim, gamedll);
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
    void LevelInit(char const* /*pMapName*/) override {}
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
