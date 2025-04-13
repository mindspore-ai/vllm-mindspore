from vllm_mindspore.config import stateless_destroy_socket_process_group

def shutdown(self):
    super(self.__class__, self).shutdown()
    if dp_group := getattr(self, "dp_group", None):
        stateless_destroy_socket_process_group(dp_group)
