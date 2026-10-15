"""Versioned benchmark bundles with traversal, link and conflict rejection."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def pack(destination, root=Path("runs"), review=False):
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if review:
        paths = [p for p in paths if p.name in ("screen_gate.json", "workflow_status.json", "audit.json",
                 "batch_audit.json", "summary.csv", "audited_summary.csv") or "figs" in p.parts]
    else:
        paths = [p for p in paths if p != root / "state.json"]
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for path in paths:
            if path.is_symlink():
                raise ValueError("Do not publish symbolic links")
            archive.add(path, arcname="runs/"+path.relative_to(root).as_posix(), recursive=False)
    if Path(destination).stat().st_size >= 2**31:
        raise ValueError("Bundle exceeds the release-asset limit")


def extract(source, destination=None):
    destination = Path.cwd() if destination is None else Path(destination)
    destination = destination.resolve()
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            name = PurePosixPath(member.name)
            target = (destination / member.name).resolve()
            if (not member.isfile() or name.is_absolute() or ".." in name.parts or
                not name.parts or name.parts[0] != "runs" or "\\" in member.name or
                destination not in target.parents or target in seen):
                raise ValueError("Unsafe or duplicated archive member")
            seen.add(target)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as incoming:
                if target.exists():
                    value = hashlib.sha256()
                    for block in iter(lambda: incoming.read(1048576), b""):
                        value.update(block)
                    if digest(target) != value.hexdigest():
                        raise ValueError("Conflicting saved output; inspect, do not overwrite")
                else:
                    with target.open("xb") as handle:
                        shutil.copyfileobj(incoming, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "extract", "review"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.path)
    else:
        pack(args.path, review=args.command == "review")
