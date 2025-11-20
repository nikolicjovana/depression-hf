from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Container for key project directories."""

    root: Path
    data: Path
    outputs: Path
    figures: Path
    models: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        data_dir = (root / "data").resolve()
        outputs_dir = (root / "outputs").resolve()
        figures_dir = outputs_dir / "figures"
        models_dir = outputs_dir / "models"
        outputs_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            data=data_dir,
            outputs=outputs_dir,
            figures=figures_dir,
            models=models_dir,
        )


def get_project_paths() -> ProjectPaths:
    """Return lazily initialised project paths."""

    root = Path(__file__).resolve().parents[2]
    return ProjectPaths.from_root(root)

