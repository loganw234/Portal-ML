"""Valve Map Format (.vmf) KeyValues brush & entity builder.

Generates valid plain-text Source Engine .vmf geometry compilable by
vbsp -> vvis -> vrad (Portal 1 / Source 2007 and Portal 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VMFSide:
    side_id: int
    plane: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    material: str
    uaxis: str
    vaxis: str
    lightmapscale: int = 16

    def to_vmf(self, indent: str = "\t\t") -> str:
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = self.plane
        p_str = (
            f"({x1:g} {y1:g} {z1:g}) "
            f"({x2:g} {y2:g} {z2:g}) "
            f"({x3:g} {y3:g} {z3:g})"
        )
        lines = [
            f"{indent}side",
            f"{indent}{{",
            f'{indent}\t"id" "{self.side_id}"',
            f'{indent}\t"plane" "{p_str}"',
            f'{indent}\t"material" "{self.material}"',
            f'{indent}\t"uaxis" "{self.uaxis}"',
            f'{indent}\t"vaxis" "{self.vaxis}"',
            f'{indent}\t"rotation" "0"',
            f'{indent}\t"lightmapscale" "{self.lightmapscale}"',
            f'{indent}\t"smoothing_groups" "0"',
            f"{indent}}}",
        ]
        return "\n".join(lines)


@dataclass(slots=True)
class VMFBrush:
    solid_id: int
    sides: list[VMFSide]

    def to_vmf(self, indent: str = "\t") -> str:
        lines = [
            f"{indent}solid",
            f"{indent}{{",
            f'{indent}\t"id" "{self.solid_id}"',
        ]
        for s in self.sides:
            lines.append(s.to_vmf(indent + "\t"))
        lines.append(f"{indent}}}")
        return "\n".join(lines)


@dataclass(slots=True)
class VMFEntity:
    entity_id: int
    classname: str
    properties: dict[str, str] = field(default_factory=dict)
    connections: list[tuple[str, str]] = field(default_factory=list)
    solids: list[VMFBrush] = field(default_factory=list)

    def to_vmf(self) -> str:
        lines = [
            "entity",
            "{",
            f'\t"id" "{self.entity_id}"',
            f'\t"classname" "{self.classname}"',
        ]
        for key, val in self.properties.items():
            lines.append(f'\t"{key}" "{val}"')
        if self.connections:
            lines.append("\tconnections")
            lines.append("\t{")
            for output_name, target_spec in self.connections:
                lines.append(f'\t\t"{output_name}" "{target_spec}"')
            lines.append("\t}")
        for solid in self.solids:
            lines.append(solid.to_vmf("\t"))
        lines.append("}")
        return "\n".join(lines)


class _IDCounter:
    def __init__(self, start: int = 1) -> None:
        self._next = start

    def take(self) -> int:
        val = self._next
        self._next += 1
        return val


def make_box_brush(
    ids: _IDCounter,
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    material: str = "METAL/METALWALL048B",
    face_materials: dict[str, str] | None = None,
) -> VMFBrush:
    """Create a valid 6-sided Source axis-aligned brush with outward-facing normals.

    `face_materials` may override any of: 'top', 'bottom', 'west', 'east', 'south', 'north'.
    """
    fm = face_materials or {}
    x0, y0, z0 = min_x, min_y, min_z
    x1, y1, z1 = max_x, max_y, max_z

    # Valve Hammer plane winding (clockwise when viewed from outside the brush)
    faces = [
        (
            "top",
            ((x0, y1, z1), (x1, y1, z1), (x1, y0, z1)),
            "[1 0 0 0] 0.25",
            "[0 -1 0 0] 0.25",
        ),
        (
            "bottom",
            ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0)),
            "[1 0 0 0] 0.25",
            "[0 -1 0 0] 0.25",
        ),
        (
            "west",
            ((x0, y1, z1), (x0, y0, z1), (x0, y0, z0)),
            "[0 1 0 0] 0.25",
            "[0 0 -1 0] 0.25",
        ),
        (
            "east",
            ((x1, y1, z0), (x1, y0, z0), (x1, y0, z1)),
            "[0 1 0 0] 0.25",
            "[0 0 -1 0] 0.25",
        ),
        (
            "north",
            ((x1, y1, z1), (x0, y1, z1), (x0, y1, z0)),
            "[1 0 0 0] 0.25",
            "[0 0 -1 0] 0.25",
        ),
        (
            "south",
            ((x1, y0, z0), (x0, y0, z0), (x0, y0, z1)),
            "[1 0 0 0] 0.25",
            "[0 0 -1 0] 0.25",
        ),
    ]

    sides = [
        VMFSide(
            side_id=ids.take(),
            plane=plane,
            material=fm.get(face_name, material),
            uaxis=uaxis,
            vaxis=vaxis,
        )
        for face_name, plane, uaxis, vaxis in faces
    ]
    return VMFBrush(solid_id=ids.take(), sides=sides)


class VMFDocument:
    """Top-level Source Engine .vmf document container."""

    def __init__(self, mapversion: int = 1) -> None:
        self.ids = _IDCounter(start=2)
        self.mapversion = mapversion
        self.world_solids: list[VMFBrush] = []
        self.entities: list[VMFEntity] = []

    def add_box(
        self,
        min_x: float,
        min_y: float,
        min_z: float,
        max_x: float,
        max_y: float,
        max_z: float,
        material: str = "METAL/METALWALL048B",
        face_materials: dict[str, str] | None = None,
    ) -> VMFBrush:
        brush = make_box_brush(
            self.ids,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            material=material,
            face_materials=face_materials,
        )
        self.world_solids.append(brush)
        return brush

    def add_entity(
        self,
        classname: str,
        properties: dict[str, str] | None = None,
        connections: list[tuple[str, str]] | None = None,
        solids: list[VMFBrush] | None = None,
    ) -> VMFEntity:
        ent = VMFEntity(
            entity_id=self.ids.take(),
            classname=classname,
            properties=dict(properties or {}),
            connections=list(connections or []),
            solids=list(solids or []),
        )
        self.entities.append(ent)
        return ent

    def add_trigger_box(
        self,
        classname: str,
        min_x: float,
        min_y: float,
        min_z: float,
        max_x: float,
        max_y: float,
        max_z: float,
        properties: dict[str, str] | None = None,
        connections: list[tuple[str, str]] | None = None,
    ) -> VMFEntity:
        brush = make_box_brush(
            self.ids,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            material="TOOLS/TOOLSTRIGGER",
        )
        return self.add_entity(
            classname=classname,
            properties=properties,
            connections=connections,
            solids=[brush],
        )

    def to_vmf(self) -> str:
        header = "\n".join(
            [
                "versioninfo",
                "{",
                '\t"editorversion" "400"',
                '\t"editorbuild" "8419"',
                f'\t"mapversion" "{self.mapversion}"',
                '\t"formatversion" "100"',
                '\t"prefab" "0"',
                "}",
                "visgroups",
                "{",
                "}",
                "viewsettings",
                "{",
                '\t"bSnapToGrid" "1"',
                '\t"bShowGrid" "1"',
                '\t"bShowLogicalGrid" "0"',
                '\t"nGridSpacing" "16"',
                "}",
                "world",
                "{",
                '\t"id" "1"',
                f'\t"mapversion" "{self.mapversion}"',
                '\t"classname" "worldspawn"',
                '\t"skyname" "sky_day01_01"',
                '\t"maxpropscreenwidth" "-1"',
                '\t"detailvbsp" "detail.vbsp"',
                '\t"detailmaterial" "detail/detailsprites"',
            ]
        )
        world_body = "\n".join(s.to_vmf("\t") for s in self.world_solids)
        world_block = f"{header}\n{world_body}\n}}"
        ent_blocks = "\n".join(e.to_vmf() for e in self.entities)
        return f"{world_block}\n{ent_blocks}\n"
