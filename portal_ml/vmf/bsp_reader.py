"""Pure-Python Valve Source BSP (v19/v20) reader for entities and portalable surfaces.

Mirrors the C++ `BspScanner` in `source_plugin/src/bsp_scanner.cpp` so offline Python
tools and tests can validate generic map discovery against compiled `.bsp` maps
(both custom `portal_ml_5chambers.bsp` and unmodified Portal 1 campaign maps).
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from portal_ml.schema.blackboard import PortalWall, Vec3

BSP_MAGIC = b"VBSP"

# Source BSP lump indices
LUMP_ENTITIES = 0
LUMP_PLANES = 1
LUMP_TEXDATA = 2
LUMP_VERTEXES = 3
LUMP_TEXINFO = 6
LUMP_FACES = 7
LUMP_EDGES = 12
LUMP_SURFEDGES = 13
LUMP_TEXDATA_STRING_DATA = 43
LUMP_TEXDATA_STRING_TABLE = 44

# SURF_NOPORTAL (Portal 1) / SURF_NODRAW / SURF_SKY / SURF_HINT
SURF_SKY = 0x0004
SURF_NODRAW = 0x0080
SURF_HINT = 0x0100
SURF_SKIP = 0x0200
SURF_IGNORE_MASK = SURF_SKY | SURF_NODRAW | SURF_HINT | SURF_SKIP

ENTITY_BLOCK_RE = re.compile(r"\{([^}]*)\}")
KV_RE = re.compile(r'"([^"]+)"\s+"([^"]*)"')


@dataclass(slots=True)
class BspEntity:
    classname: str
    targetname: str = ""
    origin: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    model: str = ""
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class BspSurface:
    id: str
    pos: Vec3
    normal: Vec3
    size: tuple[float, float]
    material: str
    orientation: str  # "wall" | "floor" | "ceiling" | "angled"

    def to_portal_wall(self, zone_id: int = 0) -> PortalWall:
        return PortalWall(
            id=self.id,
            pos=self.pos,
            normal=self.normal,
            zone_id=zone_id,
            size=self.size,
        )


@dataclass(slots=True)
class BspScanResult:
    map_name: str
    bsp_version: int
    entities: list[BspEntity] = field(default_factory=list)
    portal_surfaces: list[BspSurface] = field(default_factory=list)


def _parse_vec3(raw: str) -> Vec3:
    parts = raw.split()
    if len(parts) < 3:
        return Vec3(0.0, 0.0, 0.0)
    try:
        return Vec3(float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return Vec3(0.0, 0.0, 0.0)


def _is_portalable_material(mat: str) -> bool:
    low = mat.lower()
    if "noportal" in low or "overlay" in low or "decal" in low or "tool" in low:
        return False
    return "concrete/" in low or "concrete_modular" in low or "observationwall" in low


def _classify_orientation(normal: Vec3) -> str:
    az = normal.z
    if az >= 0.85:
        return "floor"
    if az <= -0.85:
        return "ceiling"
    if abs(az) <= 0.25:
        return "wall"
    return "angled"


def scan_bsp_bytes(data: bytes, map_name: str = "unknown", min_extent: float = 56.0) -> BspScanResult:
    """Parse entities and portalable concrete surfaces from a Source BSP v19/v20 buffer."""
    if len(data) < 1032 or data[:4] != BSP_MAGIC:
        raise ValueError("Invalid Source BSP header (expected VBSP magic)")
    version = struct.unpack_from("<i", data, 4)[0]
    lumps = [struct.unpack_from("<iiii", data, 8 + i * 16) for i in range(64)]

    def lump_slice(idx: int) -> memoryview:
        off, length, _, _ = lumps[idx]
        if off < 0 or length < 0 or off + length > len(data):
            return memoryview(b"")
        return memoryview(data)[off : off + length]

    # 1. Entities (Lump 0)
    ent_text = bytes(lump_slice(LUMP_ENTITIES)).decode("latin-1", errors="ignore")
    entities: list[BspEntity] = []
    for block in ENTITY_BLOCK_RE.findall(ent_text):
        kv = dict(KV_RE.findall(block))
        cls = kv.get("classname", "")
        if not cls:
            continue
        ang_v = _parse_vec3(kv.get("angles", "0 0 0"))
        entities.append(
            BspEntity(
                classname=cls,
                targetname=kv.get("targetname", ""),
                origin=_parse_vec3(kv.get("origin", "0 0 0")),
                angles=(ang_v.x, ang_v.y, ang_v.z),
                model=kv.get("model", ""),
                raw=kv,
            )
        )

    # 2. TexData string table (Lumps 43 & 44) and TexData (Lump 2)
    str_data = bytes(lump_slice(LUMP_TEXDATA_STRING_DATA))
    str_tbl_mv = lump_slice(LUMP_TEXDATA_STRING_TABLE)
    n_strings = len(str_tbl_mv) // 4
    str_offsets = [struct.unpack_from("<i", str_tbl_mv, i * 4)[0] for i in range(n_strings)]

    def get_string(idx: int) -> str:
        if idx < 0 or idx >= len(str_offsets):
            return ""
        start = str_offsets[idx]
        if start < 0 or start >= len(str_data):
            return ""
        end = str_data.find(b"\x00", start)
        if end < 0:
            end = len(str_data)
        return str_data[start:end].decode("latin-1", errors="ignore")

    texdata_mv = lump_slice(LUMP_TEXDATA)
    n_texdata = len(texdata_mv) // 32
    texdata_names: list[str] = []
    for i in range(n_texdata):
        name_id = struct.unpack_from("<i", texdata_mv, i * 32 + 12)[0]
        texdata_names.append(get_string(name_id))

    # 3. TexInfo (Lump 6)
    texinfo_mv = lump_slice(LUMP_TEXINFO)
    n_texinfo = len(texinfo_mv) // 72
    texinfo_list: list[tuple[int, int]] = []
    for i in range(n_texinfo):
        flags, td_idx = struct.unpack_from("<ii", texinfo_mv, i * 72 + 64)
        texinfo_list.append((flags, td_idx))

    # 4. Planes (Lump 1), Vertexes (Lump 3), Edges (Lump 12), SurfEdges (Lump 13)
    planes_mv = lump_slice(LUMP_PLANES)
    n_planes = len(planes_mv) // 20
    planes = [struct.unpack_from("<fff", planes_mv, i * 20) for i in range(n_planes)]

    verts_mv = lump_slice(LUMP_VERTEXES)
    n_verts = len(verts_mv) // 12
    verts = [struct.unpack_from("<fff", verts_mv, i * 12) for i in range(n_verts)]

    edges_mv = lump_slice(LUMP_EDGES)
    n_edges = len(edges_mv) // 4
    edges = [struct.unpack_from("<HH", edges_mv, i * 4) for i in range(n_edges)]

    surfedges_mv = lump_slice(LUMP_SURFEDGES)
    n_surfedges = len(surfedges_mv) // 4
    surfedges = [struct.unpack_from("<i", surfedges_mv, i * 4)[0] for i in range(n_surfedges)]

    # 5. Faces (Lump 7): filter portalable concrete faces of usable size
    faces_mv = lump_slice(LUMP_FACES)
    n_faces = len(faces_mv) // 56
    surfaces: list[BspSurface] = []
    surf_idx = 0
    for i in range(n_faces):
        planenum, side, _on_node, firstedge, numedges, texinfo_idx = struct.unpack_from(
            "<HBBihh", faces_mv, i * 56
        )
        if texinfo_idx < 0 or texinfo_idx >= n_texinfo or planenum >= n_planes:
            continue
        flags, td_idx = texinfo_list[texinfo_idx]
        if flags & SURF_IGNORE_MASK:
            continue
        if td_idx < 0 or td_idx >= len(texdata_names):
            continue
        mat = texdata_names[td_idx]
        if not _is_portalable_material(mat):
            continue
        if numedges < 3 or firstedge < 0 or firstedge + numedges > n_surfedges:
            continue

        nx, ny, nz = planes[planenum]
        normal = Vec3(nx, ny, nz)

        # Gather face polygon vertices
        pts: list[tuple[float, float, float]] = []
        for k in range(numedges):
            se = surfedges[firstedge + k]
            if se >= 0 and se < n_edges:
                vi = edges[se][0]
            elif se < 0 and (-se) < n_edges:
                vi = edges[-se][1]
            else:
                continue
            if 0 <= vi < n_verts:
                pts.append(verts[vi])
        if len(pts) < 3:
            continue

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)

        # Build orthonormal tangent basis (u, v) on the face plane to measure width/height
        if abs(nz) < 0.9:
            ux, uy, uz = -ny, nx, 0.0
        else:
            ux, uy, uz = 1.0, 0.0, 0.0
        ulen = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / ulen, uy / ulen, uz / ulen
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        u_vals = [(p[0] - cx) * ux + (p[1] - cy) * uy + (p[2] - cz) * uz for p in pts]
        v_vals = [(p[0] - cx) * vx + (p[1] - cy) * vy + (p[2] - cz) * vz for p in pts]
        width = max(u_vals) - min(u_vals)
        height = max(v_vals) - min(v_vals)
        if width < min_extent or height < min_extent:
            continue

        surfaces.append(
            BspSurface(
                id=f"bsp_surf_{surf_idx}",
                pos=Vec3(round(cx, 2), round(cy, 2), round(cz, 2)),
                normal=Vec3(round(nx, 4), round(ny, 4), round(nz, 4)),
                size=(round(width, 1), round(height, 1)),
                material=mat,
                orientation=_classify_orientation(normal),
            )
        )
        surf_idx += 1

    return BspScanResult(
        map_name=map_name,
        bsp_version=version,
        entities=entities,
        portal_surfaces=surfaces,
    )


def scan_bsp_file(path: str | Path, min_extent: float = 56.0) -> BspScanResult:
    p = Path(path)
    return scan_bsp_bytes(p.read_bytes(), map_name=p.stem, min_extent=min_extent)
