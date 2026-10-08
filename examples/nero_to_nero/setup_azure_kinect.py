"""在当前 Python 环境安装 Azure Kinect 依赖；SDK 放用户目录，不修改系统库。"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

PACKAGES = {
    "libk4a1.4": "1f090b65f696f78739507d0f55f3280d5df6330401b825616ef9eca0125ba8b7",
    "libk4a1.4-dev": "f7e694eae34c4b8216f249f0bf23eb0f3fd9ea00538d41b214c9e8ac2da2b3bc",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="下载微软 SDK 1.4.2 并安装 pyk4a 1.5.0")
    args = parser.parse_args()
    if not args.install:
        parser.error("确认当前 Python 环境后，显式传入 --install")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("此安装入口仅适用于 Linux x86_64")
    root = Path.home() / ".local/share/nero/azure-kinect-sdk-1.4.2"
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in PACKAGES.items():
        filename = f"{name}_1.4.2_amd64.deb"
        package = root / filename
        if not package.exists():
            url = f"https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/{name}/{filename}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"{filename} SHA256 不符，未安装")
            package.write_bytes(data)
        if hashlib.sha256(package.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"{package} SHA256 不符，请检查下载文件")
        subprocess.run(["dpkg-deb", "-x", str(package), str(root)], check=True)
    libraries = root / "usr/lib/x86_64-linux-gnu"
    env = os.environ.copy()
    env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -I{root / 'usr/include'}"
    env["LDFLAGS"] = f"{env.get('LDFLAGS', '')} -L{libraries} -Wl,-rpath,{libraries}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-binary=pyk4a",
            "--no-cache-dir",
            "--force-reinstall",
            "pyk4a==1.5.0",
        ],
        env=env,
        check=True,
    )
    subprocess.run([sys.executable, "-c", "import pyk4a; print('Azure Kinect SDK 导入通过')"], check=True)
    print(f"SDK: {root}\nPython: {sys.executable}")
    print("USB 权限规则与真机冒烟命令见 README_PLATFORM.md 的 Azure Kinect 小节。")


if __name__ == "__main__":
    main()
