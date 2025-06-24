from vllm.config import VllmConfig

from vllm_mindspore.config import stateless_destroy_socket_process_group

def _init_data_parallel(self, vllm_config: VllmConfig):
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    dp_size = vllm_config.parallel_config.data_parallel_size
    local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

    assert dp_size > 1
    assert 0 <= local_dp_rank <= dp_rank < dp_size

    self.dp_rank = dp_rank
    self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()
    self.current_wave = 0

def shutdown(self):
    super(self.__class__, self).shutdown()
    if dp_group := getattr(self, "dp_group", None):
        stateless_destroy_socket_process_group(dp_group)
