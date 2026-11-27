#include "bsp_scanner.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace portal_ml {

namespace {

constexpr int kLumpEntities = 0;
constexpr int kLumpPlanes = 1;
constexpr int kLumpTexData = 2;
constexpr int kLumpVertexes = 3;
constexpr int kLumpTexInfo = 6;
constexpr int kLumpFaces = 7;
constexpr int kLumpEdges = 12;
constexpr int kLumpSurfEdges = 13;
constexpr int kLumpModels = 14;
constexpr int kLumpTexDataStringData = 43;
constexpr int kLumpTexDataStringTable = 44;

constexpr int32_t kSurfIgnoreMask = 0x0004 | 0x0080 | 0x0100 | 0x0200;

#pragma pack(push, 1)
struct LumpEntry {
    int32_t fileofs;
    int32_t filelen;
    int32_t version;
    char fourcc[4];
};

struct DPlane {
    float nx, ny, nz;
    float dist;
    int32_t type;
};

struct DTexData {
    float reflectivity[3];
    int32_t name_string_table_id;
    int32_t width;
    int32_t height;
    int32_t view_width;
    int32_t view_height;
};

struct DModel {
    float mins[3];
    float maxs[3];
    float origin[3];
    int32_t headnode;
    int32_t firstface;
    int32_t numfaces;
};

struct DVertex {
    float x, y, z;
};

struct DTexInfo {
    float texture_vecs[2][4];
    float lightmap_vecs[2][4];
    int32_t flags;
    int32_t texdata;
};

struct DFace {
    uint16_t planenum;
    uint8_t side;
    uint8_t on_node;
    int32_t firstedge;
    int16_t numedges;
    int16_t texinfo;
    int16_t dispinfo;
    int16_t surface_fog_volume_id;
    uint8_t styles[4];
    int32_t lightofs;
    float area;
    int32_t lightmap_mins[2];
    int32_t lightmap_size[2];
    int32_t orig_face;
    uint16_t num_prims;
    uint16_t first_prim_id;
    uint32_t smoothing_groups;
};

struct DEdge {
    uint16_t v[2];
};
#pragma pack(pop)

std::string ToLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool Contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

bool IsPortalableMaterial(const std::string& mat) {
    const std::string low = ToLower(mat);
    if (Contains(low, "noportal") || Contains(low, "overlay") ||
        Contains(low, "decal") || Contains(low, "tool")) {
        return false;
    }
    return Contains(low, "concrete/") || Contains(low, "concrete_modular") ||
           Contains(low, "observationwall");
}

Vec3 ParseVec3(const std::string& s) {
    Vec3 v{};
    std::istringstream iss(s);
    iss >> v.x >> v.y >> v.z;
    return v;
}

void ParseEntitiesLump(const char* data, size_t len, const DModel* models, size_t n_models, BspMapData& out) {
    size_t i = 0;
    bool has_spawn = false;
    int idx_cube = 0;
    int idx_btn = 0;
    int idx_ped = 0;
    int idx_exit = 0;
    int idx_elev = 0;
    int idx_fizz = 0;
    int idx_hurt = 0;
    bool saw_portalgun = false;

    while (i < len) {
        while (i < len && data[i] != '{') ++i;
        if (i >= len) break;
        ++i;  // skip '{'

        std::unordered_map<std::string, std::string> kv;
        while (i < len && data[i] != '}') {
            while (i < len && data[i] != '"' && data[i] != '}') ++i;
            if (i >= len || data[i] == '}') break;
            ++i;
            size_t kstart = i;
            while (i < len && data[i] != '"') ++i;
            std::string key(data + kstart, i - kstart);
            if (i < len) ++i;

            while (i < len && data[i] != '"' && data[i] != '}') ++i;
            if (i >= len || data[i] == '}') break;
            ++i;
            size_t vstart = i;
            while (i < len && data[i] != '"') ++i;
            std::string val(data + vstart, i - vstart);
            if (i < len) ++i;

            kv[key] = val;
        }
        if (i < len && data[i] == '}') ++i;

        const auto cls_it = kv.find("classname");
        if (cls_it == kv.end()) continue;
        const std::string& cls = cls_it->second;
        const std::string tname = kv.count("targetname") ? kv["targetname"] : "";
        const std::string model = kv.count("model") ? kv["model"] : "";
        const std::string low_model = ToLower(model);
        const std::string low_tname = ToLower(tname);
        const Vec3 org = kv.count("origin") ? ParseVec3(kv["origin"]) : Vec3{};

        if (cls == "info_player_start" && !has_spawn) {
            out.player_spawn = org;
            if (kv.count("angles")) {
                const Vec3 ang = ParseVec3(kv["angles"]);
                out.player_spawn_yaw = ang.y;
            }
            has_spawn = true;
        } else if (cls == "weapon_portalgun" && !saw_portalgun) {
            saw_portalgun = true;
            if (kv.count("CanFirePortal1")) {
                out.can_fire_blue = (kv["CanFirePortal1"] != "0");
            }
            if (kv.count("CanFirePortal2")) {
                out.can_fire_orange = (kv["CanFirePortal2"] != "0");
            }
        } else if (cls == "prop_physics" &&
                   (Contains(low_model, "metal_box") || Contains(low_model, "cube") ||
                    Contains(low_tname, "box"))) {
            const std::string id = tname.empty() ? ("bsp_cube_" + std::to_string(idx_cube++)) : tname;
            out.entities.push_back({id, "cube", cls, model, org});
        } else if (cls == "func_weight_button" ||
                   (cls == "prop_dynamic" && Contains(low_model, "button_top"))) {
            const std::string id = tname.empty() ? ("bsp_floor_btn_" + std::to_string(idx_btn++)) : tname;
            out.entities.push_back({id, "floor_button", cls, model, org});
        } else if (cls == "func_button" ||
                   (cls == "prop_dynamic" && Contains(low_model, "switch001"))) {
            const std::string id = tname.empty() ? ("bsp_pedestal_" + std::to_string(idx_ped++)) : tname;
            out.entities.push_back({id, "pedestal_button", cls, model, org});
        } else if (cls == "func_door" && Contains(low_tname, "exit") && !Contains(low_tname, "right")) {
            // Keep one door leaf per double-door pair (offset to doorway center)
            const std::string id = tname.empty() ? ("bsp_exit_" + std::to_string(idx_exit++)) : tname;
            Vec3 door_pos = org;
            if (Contains(low_tname, "left")) {
                door_pos.y += 36.0f;  // center between left & right sliding door panels
            }
            out.entities.push_back({id, "exit_door", cls, model, door_pos});
        } else if (cls == "func_tracktrain" && Contains(low_tname, "elevator")) {
            const std::string id = tname.empty() ? ("bsp_elev_" + std::to_string(idx_elev++)) : tname;
            out.entities.push_back({id, "elevator", cls, model, org});
        } else if (cls == "trigger_portal_cleanser") {
            const std::string id = tname.empty() ? ("bsp_fizzler_" + std::to_string(idx_fizz++)) : tname;
            out.entities.push_back({id, "fizzler", cls, model, org});
        } else if (cls == "trigger_hurt" && models && model.size() > 1 && model[0] == '*') {
            const int midx = std::atoi(model.c_str() + 1);
            if (midx > 0 && static_cast<size_t>(midx) < n_models) {
                const DModel& dm = models[midx];
                const std::string id = tname.empty() ? ("bsp_hurt_" + std::to_string(idx_hurt++)) : tname;
                const float dps = kv.count("damage") ? static_cast<float>(std::atof(kv["damage"].c_str())) : 100.0f;
                Vec3 mn{org.x + dm.mins[0], org.y + dm.mins[1], org.z + dm.mins[2]};
                Vec3 mx{org.x + dm.maxs[0], org.y + dm.maxs[1], org.z + dm.maxs[2]};
                out.hazards.push_back({id, "trigger_hurt", mn, mx, dps > 0.0f ? dps : 100.0f});
            }
        }
    }
}

}  // namespace

bool BspScanner::LoadFromFile(const std::string& bsp_path, const std::string& map_name,
                              BspMapData& out, float min_extent) {
    out = BspMapData{};
    out.map_name = map_name;

    std::ifstream file(bsp_path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize fsize = file.tellg();
    if (fsize < 1032) return false;
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf(static_cast<size_t>(fsize));
    if (!file.read(reinterpret_cast<char*>(buf.data()), fsize)) return false;

    if (std::memcmp(buf.data(), "VBSP", 4) != 0) return false;
    std::memcpy(&out.bsp_version, buf.data() + 4, sizeof(int32_t));

    LumpEntry lumps[64]{};
    std::memcpy(lumps, buf.data() + 8, sizeof(lumps));

    auto valid_lump = [&](int idx) -> bool {
        const auto& l = lumps[idx];
        return l.fileofs >= 0 && l.filelen >= 0 &&
               static_cast<size_t>(l.fileofs) + static_cast<size_t>(l.filelen) <= buf.size();
    };

    const DModel* models_ptr = nullptr;
    size_t n_models = 0;
    if (valid_lump(kLumpModels)) {
        models_ptr = reinterpret_cast<const DModel*>(buf.data() + lumps[kLumpModels].fileofs);
        n_models = static_cast<size_t>(lumps[kLumpModels].filelen) / sizeof(DModel);
    }

    if (valid_lump(kLumpEntities)) {
        ParseEntitiesLump(
            reinterpret_cast<const char*>(buf.data() + lumps[kLumpEntities].fileofs),
            static_cast<size_t>(lumps[kLumpEntities].filelen),
            models_ptr,
            n_models,
            out);
    }

    if (!valid_lump(kLumpPlanes) || !valid_lump(kLumpTexData) || !valid_lump(kLumpVertexes) ||
        !valid_lump(kLumpTexInfo) || !valid_lump(kLumpFaces) || !valid_lump(kLumpEdges) ||
        !valid_lump(kLumpSurfEdges) || !valid_lump(kLumpTexDataStringData) ||
        !valid_lump(kLumpTexDataStringTable)) {
        out.loaded = true;
        return true;
    }

    const char* str_data = reinterpret_cast<const char*>(buf.data() + lumps[kLumpTexDataStringData].fileofs);
    const size_t str_data_len = static_cast<size_t>(lumps[kLumpTexDataStringData].filelen);
    const auto* str_tbl = reinterpret_cast<const int32_t*>(buf.data() + lumps[kLumpTexDataStringTable].fileofs);
    const size_t n_strings = static_cast<size_t>(lumps[kLumpTexDataStringTable].filelen) / sizeof(int32_t);

    auto get_string = [&](int32_t idx) -> std::string {
        if (idx < 0 || static_cast<size_t>(idx) >= n_strings) return {};
        const int32_t off = str_tbl[idx];
        if (off < 0 || static_cast<size_t>(off) >= str_data_len) return {};
        const char* p = str_data + off;
        const size_t max_len = str_data_len - static_cast<size_t>(off);
        const void* nul = std::memchr(p, '\0', max_len);
        const size_t slen = nul ? static_cast<size_t>(static_cast<const char*>(nul) - p) : max_len;
        return std::string(p, slen);
    };

    const auto* texdata = reinterpret_cast<const DTexData*>(buf.data() + lumps[kLumpTexData].fileofs);
    const size_t n_texdata = static_cast<size_t>(lumps[kLumpTexData].filelen) / sizeof(DTexData);
    std::vector<bool> texdata_portalable(n_texdata, false);
    for (size_t i = 0; i < n_texdata; ++i) {
        texdata_portalable[i] = IsPortalableMaterial(get_string(texdata[i].name_string_table_id));
    }

    const auto* texinfo = reinterpret_cast<const DTexInfo*>(buf.data() + lumps[kLumpTexInfo].fileofs);
    const size_t n_texinfo = static_cast<size_t>(lumps[kLumpTexInfo].filelen) / sizeof(DTexInfo);

    const auto* planes = reinterpret_cast<const DPlane*>(buf.data() + lumps[kLumpPlanes].fileofs);
    const size_t n_planes = static_cast<size_t>(lumps[kLumpPlanes].filelen) / sizeof(DPlane);

    const auto* verts = reinterpret_cast<const DVertex*>(buf.data() + lumps[kLumpVertexes].fileofs);
    const size_t n_verts = static_cast<size_t>(lumps[kLumpVertexes].filelen) / sizeof(DVertex);

    const auto* edges = reinterpret_cast<const DEdge*>(buf.data() + lumps[kLumpEdges].fileofs);
    const size_t n_edges = static_cast<size_t>(lumps[kLumpEdges].filelen) / sizeof(DEdge);

    const auto* surfedges = reinterpret_cast<const int32_t*>(buf.data() + lumps[kLumpSurfEdges].fileofs);
    const size_t n_surfedges = static_cast<size_t>(lumps[kLumpSurfEdges].filelen) / sizeof(int32_t);

    const auto* faces = reinterpret_cast<const DFace*>(buf.data() + lumps[kLumpFaces].fileofs);
    const size_t n_faces = static_cast<size_t>(lumps[kLumpFaces].filelen) / sizeof(DFace);

    int surf_id = 0;
    for (size_t i = 0; i < n_faces; ++i) {
        const auto& f = faces[i];
        if (f.texinfo < 0 || static_cast<size_t>(f.texinfo) >= n_texinfo || f.planenum >= n_planes) continue;
        const auto& ti = texinfo[f.texinfo];
        if ((ti.flags & kSurfIgnoreMask) != 0) continue;
        if (ti.texdata < 0 || static_cast<size_t>(ti.texdata) >= n_texdata) continue;
        if (!texdata_portalable[ti.texdata]) continue;
        if (f.numedges < 3 || f.firstedge < 0 ||
            static_cast<size_t>(f.firstedge) + static_cast<size_t>(f.numedges) > n_surfedges) {
            continue;
        }

        const auto& pl = planes[f.planenum];
        const float nx = pl.nx;
        const float ny = pl.ny;
        const float nz = pl.nz;

        std::vector<DVertex> pts;
        pts.reserve(static_cast<size_t>(f.numedges));
        for (int k = 0; k < f.numedges; ++k) {
            const int32_t se = surfedges[f.firstedge + k];
            uint16_t vi = 0;
            if (se >= 0 && static_cast<size_t>(se) < n_edges) {
                vi = edges[se].v[0];
            } else if (se < 0 && static_cast<size_t>(-se) < n_edges) {
                vi = edges[-se].v[1];
            } else {
                continue;
            }
            if (vi < n_verts) pts.push_back(verts[vi]);
        }
        if (pts.size() < 3) continue;

        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        for (const auto& p : pts) {
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        const float inv_n = 1.0f / static_cast<float>(pts.size());
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        float ux = (std::fabs(nz) < 0.9f) ? -ny : 1.0f;
        float uy = (std::fabs(nz) < 0.9f) ? nx : 0.0f;
        float uz = 0.0f;
        const float ulen = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (ulen > 1e-5f) {
            ux /= ulen;
            uy /= ulen;
            uz /= ulen;
        }
        const float vx = ny * uz - nz * uy;
        const float vy = nz * ux - nx * uz;
        const float vz = nx * uy - ny * ux;

        float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
        for (const auto& p : pts) {
            const float dx = p.x - cx;
            const float dy = p.y - cy;
            const float dz = p.z - cz;
            const float u = dx * ux + dy * uy + dz * uz;
            const float v = dx * vx + dy * vy + dz * vz;
            umin = std::min(umin, u);
            umax = std::max(umax, u);
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        const float width = umax - umin;
        const float height = vmax - vmin;
        if (width < min_extent || height < min_extent) continue;

        out.portal_surfaces.push_back({
            "bsp_surf_" + std::to_string(surf_id++),
            {cx, cy, cz},
            {nx, ny, nz},
            0,
            width,
            height,
        });
    }

    out.loaded = true;
    return true;
}

}  // namespace portal_ml
