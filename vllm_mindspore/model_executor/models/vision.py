from mindspore import Tensor, mint

def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: Tensor,
    grid_ws: Tensor,
) -> Tensor:
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx].item() // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx].item() // spatial_merge_size
    h_index = (
        mint.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(len(t_index), -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        mint.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(len(t_index), llm_grid_h, -1)
        .flatten()
    )
    t_index_tensor = (
        Tensor(t_index)
        .view(-1, 1)
        .expand(-1, llm_grid_h * llm_grid_w)
        .long()
        .flatten()
    )
    _llm_pos_ids = mint.stack([t_index_tensor, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = mint.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids
