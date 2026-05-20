from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    medulla_dir: Path = field(default_factory=lambda: Path.home() / ".medulla")

    @property
    def db_path(self) -> Path:
        return self.medulla_dir / "medulla.db"

    @property
    def wiki_path(self) -> Path:
        return self.medulla_dir / "wiki"


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
