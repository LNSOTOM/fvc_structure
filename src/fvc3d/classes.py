"""Class schemes: which labels the method transfers and how it treats them.

The paper's five rangeland growth forms are the default
(:data:`PAPER_CLASSES`). Define your own with :class:`ClassScheme`, in Python
or YAML (see ``configs/classes.yaml``)::

    from fvc3d import ClassScheme, transfer_labels

    scheme = ClassScheme(
        classes={1: "ground", 2: "wood", 3: "leaf"},
        height_ranges={1: (0.0, 0.8), 2: (0.4, 3.0), 3: (1.5, 8.0)},
        crown_class="leaf",
    )
    transfer_labels("labels.tif", "lidar.laz", "out.laz", classes=scheme)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping


def _norm(name) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class ClassScheme:
    """A set of classes and the rules the method applies to them.

    Attributes
    ----------
    classes:
        ``{code: name}``. Codes 1–255 are written to the LAS ``classification``
        field; 0 means unlabelled. Names are used in reports and ``class_map``.
    height_ranges:
        ``{code: (low, high)}`` in metres above ground. Used when a tile has
        fewer than 20 labelled points of a class to estimate its own range,
        and for the height-plausibility report.
    crown_class:
        Class whose patches are split into individual crowns and given a
        ``treeID`` (paper: mallee). ``None`` skips crown detection, polar
        refinement and the crown rules.
    merge:
        ``{from_code: to_code}``. Points of ``from_code`` higher than
        ``merge_65_to_68_znorm`` become ``to_code``; lower ones become
        unlabelled (paper: woody debris → mallee).
    promote_unlabelled:
        Turn unlabelled points at or above ``promote_unlabelled_znorm`` into
        ``crown_class``.
    raster_map:
        Optional ``{raster_value: code}`` for ``class_map="paper"``-style
        presets: how this scheme's published rasters encode the classes.
    output_classes, output_map:
        Optional relabelling applied after the method, e.g. to FOR-instanceV2's
        ground/wood/leaf (the classes ForestFormer3D is trained on). ``output_classes`` is ``{code: name}`` for the output;
        ``output_map`` is ``{class: output_class}`` (names or codes). Classes not
        in ``output_map`` become unlabelled. ``treeID`` is unchanged.
    """

    classes: Mapping[int, str]
    height_ranges: Mapping[int, tuple[float, float]]
    crown_class: int | None = None
    merge: Mapping[int, int] = field(default_factory=dict)
    promote_unlabelled: bool = False
    raster_map: Mapping[int, int] | None = None
    output_classes: Mapping[int, str] | None = None
    output_map: Mapping[int, int] | None = None

    def __post_init__(self):
        classes = {int(k): str(v).strip().lower().replace(" ", "_").replace("-", "_") for k, v in self.classes.items()}
        if not classes:
            raise ValueError("A class scheme needs at least one class")
        bad = [c for c in classes if not 1 <= c <= 255]
        if bad:
            raise ValueError(f"Class codes must be 1–255 (0 means unlabelled): {bad}")
        if len(set(classes.values())) != len(classes):
            raise ValueError("Class names must be unique")
        object.__setattr__(self, "classes", dict(sorted(classes.items())))

        ranges = {int(k): (float(v[0]), float(v[1])) for k, v in self.height_ranges.items()}
        missing = sorted(set(classes) - set(ranges))
        if missing:
            raise ValueError(f"height_ranges is missing classes {missing}")
        for code, (lo, hi) in ranges.items():
            if code not in classes:
                raise ValueError(f"height_ranges has unknown class {code}")
            if lo > hi:
                raise ValueError(f"height range for class {code} has low > high: {(lo, hi)}")
        object.__setattr__(self, "height_ranges", {c: ranges[c] for c in classes})

        crown = self.code(self.crown_class) if self.crown_class is not None else None
        object.__setattr__(self, "crown_class", crown)
        merge = {self.code(k): self.code(v) for k, v in self.merge.items()}
        object.__setattr__(self, "merge", merge)
        if self.promote_unlabelled and crown is None:
            raise ValueError("promote_unlabelled needs a crown_class")
        if self.raster_map is not None:
            rmap = {int(k): self.code(v) for k, v in self.raster_map.items()}
            object.__setattr__(self, "raster_map", rmap)

        if (self.output_classes is None) != (self.output_map is None):
            raise ValueError("output_classes and output_map must be given together")
        if self.output_classes is not None:
            out = {int(k): _norm(v) for k, v in self.output_classes.items()}
            bad = [c for c in out if not 1 <= c <= 255]
            if bad or len(set(out.values())) != len(out):
                raise ValueError("output_classes needs unique names and codes 1–255")
            by_name = {v: k for k, v in out.items()}

            def out_code(v):
                if isinstance(v, str) and not v.strip().isdigit():
                    if _norm(v) not in by_name:
                        raise ValueError(f"Unknown output class {v!r}; output_classes has {sorted(by_name)}")
                    return by_name[_norm(v)]
                if int(v) not in out:
                    raise ValueError(f"Output code {v} is not in output_classes")
                return int(v)

            object.__setattr__(self, "output_classes", dict(sorted(out.items())))
            object.__setattr__(self, "output_map", {self.code(k): out_code(v) for k, v in self.output_map.items()})

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(self.classes)

    @property
    def names(self) -> dict[str, int]:
        return {name: code for code, name in self.classes.items()}

    def code(self, value) -> int:
        """Class code for a code or name."""
        if isinstance(value, str) and not value.strip().isdigit():
            key = value.strip().lower().replace(" ", "_").replace("-", "_")
            if key not in self.names:
                raise ValueError(f"Unknown class name {value!r}; this scheme has {sorted(self.names)}")
            return self.names[key]
        code = int(value)
        if code not in self.classes:
            raise ValueError(f"Class code {code} is not in this scheme {self.codes}")
        return code

    def name(self, code: int) -> str:
        return self.classes.get(int(code), str(code))

    @property
    def final_classes(self) -> dict[int, str]:
        """Classes in the output: ``output_classes`` if set, else ``classes``."""
        return dict(self.output_classes if self.output_classes is not None else self.classes)

    @property
    def final_crown_class(self) -> int | None:
        if self.crown_class is None or self.output_map is None:
            return self.crown_class
        return self.output_map.get(self.crown_class)

    def to_output(self, classification):
        """Relabel method classes to output classes (no-op without ``output_map``)."""
        import numpy as np

        cls = np.asarray(classification)
        if self.output_map is None:
            return cls
        lookup = np.zeros(256, dtype=np.uint8)
        for src, dst in self.output_map.items():
            lookup[src] = dst
        return lookup[cls.astype(np.int64)]

    def with_changes(self, **changes) -> "ClassScheme":
        """Copy with some fields replaced, e.g. ``PAPER_CLASSES.with_changes(crown_class="shrub")``."""
        return replace(self, **changes)

    @classmethod
    def from_any(cls, value) -> "ClassScheme":
        """Build from ``None``/``"paper"``, a ``ClassScheme``, a dict, or a YAML file path."""
        if value is None or (isinstance(value, str) and value.lower() == "paper"):
            return PAPER_CLASSES
        if isinstance(value, str) and "".join(ch for ch in value.lower() if ch.isalnum()) == "forinstancev2":
            return FORINSTANCEV2_CLASSES
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            import yaml

            with open(value) as fh:
                value = yaml.safe_load(fh) or {}
        if not isinstance(value, Mapping):
            raise TypeError(f'classes must be "paper", a ClassScheme, a dict or a YAML path, not {type(value).__name__}')
        known = {"classes", "height_ranges", "crown_class", "merge", "promote_unlabelled", "raster_map",
                 "output_classes", "output_map"}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown class-scheme keys {unknown}; valid keys: {sorted(known)}")
        return cls(**value)

    def to_dict(self) -> dict:
        return {
            "classes": dict(self.classes),
            "height_ranges": {c: list(r) for c, r in self.height_ranges.items()},
            "crown_class": self.crown_class,
            "merge": dict(self.merge),
            "promote_unlabelled": self.promote_unlabelled,
            "raster_map": None if self.raster_map is None else dict(self.raster_map),
            "output_classes": None if self.output_classes is None else dict(self.output_classes),
            "output_map": None if self.output_map is None else dict(self.output_map),
        }


#: The paper's five rangeland growth forms. Codes 64–68 are the LAS classes;
#: the published label rasters code them 0–4 (``class_map="paper"``).
PAPER_CLASSES = ClassScheme(
    classes={64: "forb", 65: "woody_debris", 66: "hummock", 67: "shrub", 68: "mallee"},
    height_ranges={
        64: (0.00, 0.40),
        65: (0.05, 5.00),
        66: (0.00, 0.80),
        67: (0.40, 3.00),
        68: (1.50, 8.00),
    },
    crown_class=68,
    merge={65: 68},
    promote_unlabelled=True,
    raster_map={0: 64, 1: 65, 2: 66, 3: 67, 4: 68},
)


#: The paper's method with FOR-instanceV2's semantic classes as output, as in the
#: paper's cross-domain experiments: hummock and forb → ground, shrub → wood,
#: mallee → leaf (woody debris is already merged into mallee or dropped by
#: Stage 5). Codes 1–3; ForestFormer3D model IDs are code − 1 (0 ground,
#: 1 wood, 2 leaf). For rangelands these are approximate transfer targets, not
#: a literal taxonomy. Select with ``classes="FOR-instanceV2"``.
FORINSTANCEV2_CLASSES = PAPER_CLASSES.with_changes(
    output_classes={1: "ground", 2: "wood", 3: "leaf"},
    output_map={"forb": "ground", "hummock": "ground", "shrub": "wood", "mallee": "leaf"},
)
