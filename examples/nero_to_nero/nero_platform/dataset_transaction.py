"""少量数据集文件的可恢复替换。原件保留到提交完成，不依赖 Runtime 或硬件。

多文件替换并非原子事务：journal 存在时读取/续录必须等待恢复。
prepared 可重复回滚，committed 只清理原件；进程中断后采用相同规则。
"""

import contextlib
import json
import os
import shutil
from pathlib import Path

JOURNAL = ".nero-delete-transaction"


def pending(root: Path) -> bool:
    return (root / JOURNAL).exists()


def _sync(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mark(work: Path, manifest: dict) -> None:
    tmp = work / "manifest.tmp"
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(work / "manifest.json")
    _sync(work)


def _target(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("删除日志包含非法路径，保留原件等待检查")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("删除日志指向数据集外部，保留原件等待检查")
    return path


def recover(root: Path) -> None:
    """恢复中断操作；发生任何恢复错误时保留 journal，禁止后续写入。"""
    work = root / JOURNAL
    if not work.exists():
        return
    journal = work / "manifest.json"
    if journal.exists():
        manifest = json.loads(journal.read_text(encoding="utf-8"))
        if manifest["state"] not in ("prepared", "committed"):
            raise ValueError("无法识别删除日志，请保留目录并检查")
        if manifest["state"] == "prepared":
            for entry in manifest["files"]:
                target = _target(root, entry["path"])
                if entry["existed"]:
                    original = work / "original" / entry["path"]
                    # 硬链接原件始终保留，恢复过程自身被中断后也能再次执行。
                    tmp = work / "restore.tmp"
                    tmp.unlink(missing_ok=True)
                    os.link(original, tmp)
                    os.replace(tmp, target)
                else:
                    target.unlink(missing_ok=True)
                _sync(target.parent)
            # 回滚已完成；后续清理即使中断也不应再次依赖原件。
            _mark(work, {**manifest, "state": "committed"})
    # 没有 manifest 表示原件尚未开始修改。
    shutil.rmtree(work)
    _sync(root)


def replace_files(root: Path, replacements: dict[str, Path | None]) -> None:
    """替换为预先生成的文件，None 表示删除。调用者已完成全部校验。

    暂存原件使用同盘硬链接，视频不复制、不重编码；提交只 rename/unlink，
    不会原地改写原件。成功后清理，失败则恢复。此暂存不能代替独立备份。
    """
    work = root / JOURNAL
    work.mkdir()  # 不覆盖未恢复事务。
    committed = False
    try:
        entries = []
        for relative, replacement in replacements.items():
            target = _target(root, relative)
            existed = target.is_file()
            entries.append({"path": relative, "existed": existed})
            if existed:
                backup = work / "original" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, backup)
            if replacement is not None:
                with replacement.open("rb") as stream:
                    os.fsync(stream.fileno())
        # 先持久化所有备份目录，再写准备完成标记，最后开始替换正式文件。
        for directory in sorted((p for p in work.rglob("*") if p.is_dir()), reverse=True):
            _sync(directory)
        manifest = {"state": "prepared", "files": entries}
        _mark(work, manifest)
        _sync(root)
        for relative, replacement in replacements.items():
            target = _target(root, relative)
            if replacement is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(replacement, target)
            _sync(target.parent)
        _mark(work, {**manifest, "state": "committed"})
        committed = True
    except Exception as exc:
        # 提交标记已替换，但目录 fsync/后续清理报错时，数据实际已提交。
        # 不能把它报告为回滚失败，让用户误以为需要再次删除。
        journal = work / "manifest.json"
        try:
            committed = journal.exists() and json.loads(journal.read_text())["state"] == "committed"
        except (OSError, ValueError, KeyError):
            committed = False
        if committed:
            return
        try:
            recover(root)
        except Exception as recovery_error:
            raise RuntimeError(
                f"删除未完成且恢复失败：{recovery_error}。原件保留在 {work}；"
                "修复磁盘问题后重新点击删除以恢复，恢复完成前不能续录。"
            ) from exc
        raise RuntimeError(f"删除失败，已恢复原数据，可重新确认后重试：{exc}") from exc
    finally:
        if committed:
            # 提交后的暂存清理失败不谎报为删除失败，下次操作会再清理。
            with contextlib.suppress(OSError):
                recover(root)
