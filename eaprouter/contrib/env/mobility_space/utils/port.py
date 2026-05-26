import socket
import time
from typing import List

from eaprouter.logger import get_logger

__all__ = ["find_free_ports", "wait_for_port"]

logger = get_logger()


def find_free_ports(num_ports: int = 1) -> List[int]:
    ports: list[int] = []
    sockets = []

    for _ in range(num_ports):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ports.append(s.getsockname()[1])
        sockets.append(s)
    for s in sockets:
        s.close()
    return ports


def wait_for_port(host: str, port: int, timeout: float = 30.0, check_interval: float = 0.5) -> bool:
    """
    Wait for a port to become available (listening).
    
    Args:
        host: The host to check (e.g., "localhost" or "127.0.0.1")
        port: The port number to check
        timeout: Maximum time to wait in seconds (default: 30.0)
        check_interval: Time between checks in seconds (default: 0.5)
    
    Returns:
        True if the port becomes available within the timeout, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                result = sock.connect_ex((host, port))
                if result == 0:
                    # Port is open and listening
                    return True
        except (socket.error, socket.timeout) as e:
            # 输出报错内容
            logger.warning(f"Error: {e}")
            pass
        time.sleep(check_interval)
    return False
