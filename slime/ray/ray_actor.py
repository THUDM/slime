import ray
from slime.utils.http_utils import is_port_available


class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        address = address.strip("[]")

        # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available
        port = start_port
        while not all(is_port_available(port + i) for i in range(consecutive)):
            port += 1

        return address, port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
