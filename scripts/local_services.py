"""管理当前 macOS 本地安装的四个服务；不安装依赖，不删除数据。"""

import fcntl
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
PIDS = RUNTIME / "pids"
LOGS = RUNTIME / "logs"
PORTS = {"mongodb": 27017, "redis": 6379, "backend": 8000, "frontend": 3100}


def process(pid):
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "stat=", "-o", "command="],
        capture_output=True, text=True,
    )
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return None
    if result.returncode:
        raise RuntimeError("无法检查进程，请在本机终端中运行脚本。")
    output = result.stdout.strip()
    if not output or output.split()[0].startswith("Z"):
        return None
    return output


def cwd_of(pid):
    result = subprocess.run(
        ["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
        capture_output=True, text=True,
    )
    return next((line[1:] for line in result.stdout.splitlines() if line.startswith("n")), "")


def managed_pid(name):
    path = PIDS / f"{name}.pid"
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        raise RuntimeError(f"PID 文件损坏，请检查 {path}")
    if pid <= 1:
        raise RuntimeError(f"PID 不合法：{path}")
    command = process(pid)
    if command is None:
        path.unlink(missing_ok=True)
        return None
    cwd = cwd_of(pid)
    owned = {
        "mongodb": "mongod" in command and f"--dbpath {RUNTIME / 'data/mongodb'}" in command,
        "redis": "redis-server" in command and cwd == str(RUNTIME / "data/redis"),
        "backend": "-m uvicorn app.main:app" in command and cwd == str(ROOT),
        "frontend": str(ROOT / "frontend/node_modules/vite/bin/vite.js") in command
                    and cwd == str(ROOT / "frontend"),
    }[name]
    if not owned:
        raise RuntimeError(f"{name} 的 PID {pid} 不属于本项目，拒绝操作。请检查 {path}")
    return pid


def listening(port):
    with socket.socket() as connection:
        connection.settimeout(0.3)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def ready(name):
    if not listening(PORTS[name]):
        return False
    if name not in ("backend", "frontend"):
        return True
    url = ("http://127.0.0.1:8000/api/health" if name == "backend"
           else "http://127.0.0.1:3100/")
    try:
        # 本地就绪检查不走系统 HTTP 代理。
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def wait_ready(name, pid):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process(pid) is None:
            raise RuntimeError(f"{name} 启动后退出，请查看 {LOGS / (name + '.log')}")
        if ready(name):
            return
        time.sleep(1)
    raise RuntimeError(f"{name} 120 秒内未就绪，请查看 {LOGS / (name + '.log')}")


def stop_one(name):
    pid = managed_pid(name)
    if pid is None:
        if listening(PORTS[name]):
            raise RuntimeError(f"端口 {PORTS[name]} 被未登记的进程使用，拒绝猜测并关闭。")
        print(f"{name} 已停止", flush=True)
        return
    print(f"关闭 {name}（PID {pid}）…", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 45
    while process(pid) is not None:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{name} 未在 45 秒内退出；未强制终止，请查看日志。")
        time.sleep(0.5)
    (PIDS / f"{name}.pid").unlink(missing_ok=True)
    if listening(PORTS[name]):
        raise RuntimeError(f"{name} 已退出，但端口 {PORTS[name]} 仍被其他进程占用。")


def commands():
    mongo_candidates = sorted((RUNTIME / "mongodb").glob("*/bin/mongod"))
    if len(mongo_candidates) != 1:
        raise RuntimeError("runtime/mongodb 中必须有且仅有一套本地 MongoDB。")
    node = shutil.which("node")
    if not node:
        candidates = list((Path.home() / ".nvm/versions/node").glob("v*/bin/node"))
        candidates.sort(key=lambda p: tuple(int(x) for x in p.parents[1].name.lstrip("v").split(".")))
        node = str(candidates[-1]) if candidates else None
    required = [ROOT / ".venv/bin/python", ROOT / ".env", RUNTIME / "redis.conf",
                RUNTIME / "redis/src/redis-server", ROOT / "frontend/node_modules/vite/bin/vite.js"]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"缺少 {path}，请先完成本地安装。")
    if not node:
        raise RuntimeError("未找到 Node.js，请先安装 Node.js 22 或更新版本。")
    return {
        "mongodb": [str(mongo_candidates[0]), "--dbpath", str(RUNTIME / "data/mongodb"),
                    "--bind_ip", "127.0.0.1", "--port", "27017", "--auth",
                    "--logpath", str(LOGS / "mongodb.log"), "--logappend",
                    "--unixSocketPrefix", str(RUNTIME)],
        "redis": [str(RUNTIME / "redis/src/redis-server"), str(RUNTIME / "redis.conf"),
                  "--daemonize", "no", "--bind", "127.0.0.1", "--port", "6379",
                  "--dir", str(RUNTIME / "data/redis"), "--logfile", str(LOGS / "redis.log"),
                  "--pidfile", str(PIDS / "redis.pid")],
        "backend": [str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "app.main:app",
                    "--host", "127.0.0.1", "--port", "8000"],
        "frontend": [node, str(ROOT / "frontend/node_modules/vite/bin/vite.js"),
                     "--host", "127.0.0.1", "--port", "3100", "--strictPort"],
    }


def start():
    service_commands = commands()
    existing = {name: managed_pid(name) for name in PORTS}
    for name, pid in existing.items():
        if not pid and listening(PORTS[name]):
            raise RuntimeError(f"端口 {PORTS[name]} 已被其他进程占用，未启动任何新服务。")
    for directory in (RUNTIME / "data/mongodb", RUNTIME / "data/redis", LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": str(RUNTIME / "matplotlib"),
           "VITE_API_TARGET": "http://127.0.0.1:8000", "VITE_PORT": "3100"}
    started = []
    try:
        for name, command in service_commands.items():
            pid = existing[name]
            if pid:
                print(f"{name} 已运行（PID {pid}）", flush=True)
            else:
                with (LOGS / f"{name}.log").open("ab") as log:
                    child = subprocess.Popen(command, cwd=ROOT / "frontend" if name == "frontend" else ROOT,
                                             env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                pid = child.pid
                (PIDS / f"{name}.pid").write_text(f"{pid}\n")
                started.append(name)
                print(f"启动 {name}（PID {pid}）…", flush=True)
            wait_ready(name, pid)
        print("启动完成：http://127.0.0.1:3100\n日志目录：" + str(LOGS), flush=True)
    except BaseException:
        for name in reversed(started):
            try:
                stop_one(name)
            except Exception as error:
                print(f"回收 {name} 失败：{error}", file=sys.stderr)
        raise


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("start", "stop"):
        raise RuntimeError("用法：local_services.py start|stop")
    PIDS.mkdir(parents=True, exist_ok=True)
    with (PIDS / "services.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("另一个启动/关闭操作正在执行，请稍后再试。")
        if sys.argv[1] == "start":
            start()
        else:
            for name in reversed(PORTS):
                stop_one(name)
            print("全部服务已关闭，数据库文件已保留。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"操作未完成：{error}", file=sys.stderr)
        sys.exit(1)
