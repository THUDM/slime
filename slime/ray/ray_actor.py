import ray
from slime.utils.http_utils import is_port_available


class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(num_ports=1, start_port=50000):
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        address = address.strip("[]")

        port = start_port
        ports = []
        while len(ports) < num_ports:
            if is_port_available(port):
                ports.append(port)
            port += 1

        if num_ports == 1:
            return address, ports[0]

        return address, ports

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
