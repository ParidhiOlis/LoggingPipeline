"""
Log File Manager for Batch Processing
Handles log rotation, file management, and preparation for ClickHouse ingestion
"""

import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import logging
from dataclasses import dataclass
import shutil


@dataclass
class LogFileInfo:
    path: Path
    stage: str
    date: str
    size: int
    line_count: int
    created_at: datetime
    platform: str = "unknown"


class LogFileManager:

    def __init__(
        self,
        base_log_dir: str = "logs",
        max_file_size_mb: int = 100,
        platform_rotation_config: Dict[str, str] = None
    ):
        self.base_log_dir = Path(base_log_dir)
        self.max_file_size_mb = max_file_size_mb
        self.logger = logging.getLogger(__name__)

        self.platform_rotation_config = platform_rotation_config or {
            "sharepoint": "12hours",  # 2 times per day
            "slack": "30minutes",     # every 30 minutes
            "outlook": "30minutes",   # every 30 minutes
            "notion": "12hours",      # 2 times per day
            "default": "daily"
        }

        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_platform_directories()

    def _setup_platform_directories(self):
        platforms = ["sharepoint", "slack", "outlook", "notion"]
        stages = ["extraction", "indexing", "queries"]

        for platform in platforms:
            for stage in stages:
                (self.base_log_dir / platform / stage).mkdir(parents=True, exist_ok=True)

    def _get_rotation_timestamp(self, platform: str) -> str:
        now = datetime.now()
        freq = self.platform_rotation_config.get(platform, "daily")

        if freq == "30minutes":
            minute = (now.minute // 30) * 30
            return now.replace(minute=minute, second=0, microsecond=0).strftime("%Y-%m-%d_%H-%M")
        elif freq == "12hours":
            hour = 0 if now.hour < 12 else 12
            return now.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d_%H-%M")
        elif freq == "hourly":
            return now.strftime("%Y-%m-%d_%H")
        else:
            return now.strftime("%Y-%m-%d")

    def get_current_log_file(self, stage: str, platform: str) -> Path:
        stage_dir = self.base_log_dir / platform / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = f"{platform}_{stage}"
        timestamp = self._get_rotation_timestamp(platform)

        pattern = f"{file_prefix}_{timestamp}_*.jsonl"
        existing_files = sorted(stage_dir.glob(pattern))

        if existing_files:
            current_file = existing_files[-1]
            if current_file.stat().st_size > (self.max_file_size_mb * 1024 * 1024):
                new_file = stage_dir / f"{file_prefix}_{timestamp}_{len(existing_files)+1:03d}.jsonl"
                return new_file
            return current_file
        else:
            return stage_dir / f"{file_prefix}_{timestamp}_001.jsonl"

    def list_log_files(self, stage: Optional[str], platform: Optional[str], days_back: int = 7) -> List[LogFileInfo]:
        cutoff = datetime.now() - timedelta(days=days_back)
        search_dirs = []

        if platform:
            if stage:
                search_dirs.append(self.base_log_dir / platform / stage)
            else:
                for s in ["extraction", "indexing", "queries"]:
                    search_dirs.append(self.base_log_dir / platform / s)
        else:
            return []

        log_files = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for file in search_dir.glob("*.jsonl"):
                try:
                    date_str = next((p for p in file.stem.split('_') if len(p) == 10 and p.count('-') == 2), None)
                    if date_str:
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if file_date >= cutoff:
                            log_files.append(LogFileInfo(
                                path=file,
                                stage=stage or "unknown",
                                date=date_str,
                                size=file.stat().st_size,
                                line_count=self._count_lines(file),
                                created_at=datetime.fromtimestamp(file.stat().st_ctime),
                                platform=platform
                            ))
                except Exception:
                    continue
        return log_files

    def _count_lines(self, path: Path) -> int:
        try:
            with open(path, 'rb') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def read_log_entries(self, path: Path) -> Generator[Dict[str, Any], None, None]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            self.logger.error(f"Error reading log file {path}: {e}")

    def prepare_for_clickhouse(self, stage: str, platform: str, output_file: Path, days_back: int = 1) -> Dict[str, Any]:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        stats = {"files_processed": 0, "total_entries": 0, "valid_entries": 0, "invalid_entries": 0}

        files = self.list_log_files(stage, platform, days_back)
        with open(output_file, 'w', encoding='utf-8') as out:
            for f in files:
                stats["files_processed"] += 1
                for entry in self.read_log_entries(f.path):
                    stats["total_entries"] += 1
                    if all(k in entry for k in ["timestamp", "level", "event"]):
                        out.write(json.dumps(entry) + "\n")
                        stats["valid_entries"] += 1
                    else:
                        stats["invalid_entries"] += 1
        return stats
