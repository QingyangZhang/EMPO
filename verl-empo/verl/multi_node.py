import os
import socket
import subprocess
import sys
import time
import ray
from datetime import datetime

# 先获取当前时间并格式化为 "年-月-日_时-分" 的字符串
# 例如: '20250902_1523'
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

SHARED_DIR = "/mnt/shared-storage-user/p1-shared/zhangqingyang"
# 使用 f-string 将时间戳动态插入到文件名中
# 对于有后缀的文件，将时间戳放在后缀名前
MASTER_ADDR_FILE = os.path.join(SHARED_DIR, f"master_addr_30B_{timestamp}.txt")

# 对于没有后缀的文件，直接拼接在结尾
READY_FLAG_FILE = os.path.join(SHARED_DIR, f"ray_head_ready_30B_{timestamp}")

MASTER_PORT = 6379

COMMAND = os.environ.get('COMMAND', 'bash')
SCRIPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SCRIPT', 'echo hello world')

def log(msg):
    print(f"[multi_node_rjob.py] {msg}", file=sys.stderr, flush=True)

def get_rank():
    return int(os.environ.get("KUBEBRAIN_REPLICA", -1))

def get_world_size():
    return int(os.environ.get("KUBEBRAIN_REPLICA_TOTAL", 1))

def get_my_ip():
    return socket.gethostbyname(socket.gethostname())

def start_ray_head_node():    
    my_ip = get_my_ip()
    log(f"Starting ray head on {my_ip}...")
    with open(MASTER_ADDR_FILE, "w") as f:
        f.write(my_ip)
    subprocess.run(["ray", "start", "--head",
                    f"--port={MASTER_PORT}",
                    f"--node-ip-address={my_ip}",
                    "--num-gpus=8",
                    "--disable-usage-stats"], check=True)
    open(READY_FLAG_FILE, "w").close()
    return f"{my_ip}:{MASTER_PORT}"

def wait_for_master():
    log(f"Waiting for master address file {MASTER_ADDR_FILE}...")
    for _ in range(1200):  # timeout 20 min
        if os.path.exists(MASTER_ADDR_FILE):
            with open(MASTER_ADDR_FILE, "r") as f:
                addr = f.read().strip()
            log(f"Got master address: {addr}")
            return addr
        else:
            log(f"No master address yet...")
        time.sleep(1)
    raise RuntimeError("Timed out waiting for master address")

def connect_ray_worker(master_addr):
    my_ip = get_my_ip()
    log(f"Connecting to head node at {master_addr} from {my_ip}")
    subprocess.run(["ray", "start",
                    f"--address={master_addr}",
                    f"--node-ip-address={my_ip}",
                    "--num-gpus=8",
                    "--disable-usage-stats",
                    "--block"], check=True)

def wait_for_all_nodes(expected_nodes, timeout=600):
    ray.init(address="auto")
    log(f"Waiting for all {expected_nodes} nodes to connect...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            nodes = ray.nodes()
            alive = [n for n in nodes if n["Alive"]]
            log(f"{len(alive)} / {expected_nodes} nodes connected")
            if len(alive) >= expected_nodes:
                ray.shutdown()
                return True
        except Exception as e:
            log(f"Error checking cluster state: {e}")
        time.sleep(5)
    ray.shutdown()
    return False

def execute_entry_command():
    log(f"Executing: {COMMAND} {SCRIPT}")
    subprocess.run([COMMAND, SCRIPT], check=True)

def main():
    rank = get_rank()
    world_size = get_world_size()

    log(f"RANK={rank}, WORLD_SIZE={world_size}")

    if rank == 0:
        master_addr = start_ray_head_node()
        # 等待其他节点加入
        if wait_for_all_nodes(world_size):
            log("All nodes joined. Running main script.")
            execute_entry_command()
        else:
            log("Timeout waiting for nodes.")
            sys.exit(1)
    else:
        master_addr = wait_for_master()
        connect_ray_worker(f"{master_addr}:{MASTER_PORT}")
        

if __name__ == "__main__":
    log(f"COMMAND={COMMAND}")
    log(f"SCRIPT={SCRIPT}")
    main()