from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
DEFAULT_SIMULATION_DIR = "LiveFeed Simulation"


@dataclass
class SimulationClip:
    clip_id: str
    canonical_path: str
    relative_path: str
    display_name: str
    file_size: int
    sha1: str
    aliases: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "canonical_path": self.canonical_path,
            "relative_path": self.relative_path,
            "display_name": self.display_name,
            "file_size": self.file_size,
            "sha1": self.sha1,
            "aliases": list(self.aliases),
        }


class SimulationVideoRegistry:
    def __init__(self, simulation_root: str):
        self.simulation_root = os.path.abspath(simulation_root)
        self._clips_by_id: Dict[str, SimulationClip] = {}
        self._sorted_ids: List[str] = []
        self._errors: List[str] = []

    @staticmethod
    def _looks_like_video(path: str) -> bool:
        _, ext = os.path.splitext(path)
        return ext.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def _normalized_name(path: str) -> str:
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        lowered = name.lower()
        lowered = lowered.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
        lowered = re.sub(r"-+", "-", lowered).strip("-")
        return lowered or "clip"

    @staticmethod
    def _path_priority(relative_path: str) -> Tuple[int, int]:
        folder = relative_path.replace("\\", "/").lower()
        score = 50
        if folder == "basic loop.mp4":
            score = 0
        elif folder.startswith("t ") or folder.startswith("t"):
            score = 10
        elif folder.startswith("rein gehen/") or folder.startswith("raus gehen/"):
            score = 20
        elif folder[:1].isdigit():
            score = 30
        return (score, len(relative_path))

    @staticmethod
    def _sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha1()
        with open(path, "rb") as handle:
            while True:
                data = handle.read(chunk_size)
                if not data:
                    break
                digest.update(data)
        return digest.hexdigest()

    def refresh(self) -> Dict[str, Any]:
        self._clips_by_id = {}
        self._sorted_ids = []
        self._errors = []

        if not os.path.isdir(self.simulation_root):
            self._errors.append(f"Simulation directory not found: {self.simulation_root}")
            return self.summary()

        hash_groups: Dict[Tuple[int, str], List[str]] = {}
        for root, _, files in os.walk(self.simulation_root):
            for file_name in files:
                path = os.path.join(root, file_name)
                if not self._looks_like_video(path):
                    continue
                try:
                    file_size = os.path.getsize(path)
                    sha1_value = self._sha1_file(path)
                    key = (file_size, sha1_value)
                    hash_groups.setdefault(key, []).append(path)
                except Exception as exc:
                    self._errors.append(f"{path}: {exc}")

        used_ids: set[str] = set()
        for (file_size, sha1_value), group_paths in hash_groups.items():
            sorted_paths = sorted(group_paths, key=lambda p: self._path_priority(os.path.relpath(p, self.simulation_root)))
            canonical_path = os.path.abspath(sorted_paths[0])
            canonical_rel = os.path.relpath(canonical_path, self.simulation_root).replace("\\", "/")
            base_id = self._normalized_name(canonical_rel)
            clip_id = base_id
            idx = 2
            while clip_id in used_ids:
                clip_id = f"{base_id}-{idx}"
                idx += 1
            used_ids.add(clip_id)

            aliases = sorted({os.path.abspath(path) for path in group_paths if os.path.abspath(path) != canonical_path})
            clip = SimulationClip(
                clip_id=clip_id,
                canonical_path=canonical_path,
                relative_path=canonical_rel,
                display_name=os.path.splitext(os.path.basename(canonical_path))[0].strip() or clip_id,
                file_size=file_size,
                sha1=sha1_value,
                aliases=aliases,
            )
            self._clips_by_id[clip_id] = clip

        self._sorted_ids = sorted(self._clips_by_id.keys(), key=lambda cid: self._clips_by_id[cid].relative_path.lower())
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        total_aliases = sum(len(c.aliases) for c in self._clips_by_id.values())
        return {
            "simulation_root": self.simulation_root,
            "clips": len(self._clips_by_id),
            "duplicates": total_aliases,
            "errors": list(self._errors),
        }

    def list_clips(self) -> List[Dict[str, Any]]:
        return [self._clips_by_id[clip_id].to_dict() for clip_id in self._sorted_ids]

    def get_clip(self, clip_id: str) -> Optional[SimulationClip]:
        return self._clips_by_id.get(str(clip_id or "").strip())

    def find_clip_id_by_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        abs_path = os.path.abspath(path)
        for clip_id in self._sorted_ids:
            clip = self._clips_by_id[clip_id]
            if abs_path == clip.canonical_path:
                return clip_id
            if abs_path in clip.aliases:
                return clip_id
        return None

    def get_default_basic_loop(self) -> Optional[SimulationClip]:
        for clip_id in self._sorted_ids:
            clip = self._clips_by_id[clip_id]
            if clip.relative_path.lower() == "basic loop.mp4":
                return clip
        return None
