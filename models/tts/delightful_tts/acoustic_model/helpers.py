import torch


def average_over_durations(values: torch.Tensor, durs: torch.Tensor) -> torch.Tensor:
    r"""Function calculates the average of values over specified durations.

    Args:
    values (torch.Tensor): A 3D tensor of shape [B, 1, T_de] where B is the batch size,
                           T_de is the duration of each element in the batch. The values
                           represent some quantity that needs to be averaged over durations.
    durs (torch.Tensor): A 2D tensor of shape [B, T_en] where B is the batch size,
                         T_en is the number of elements in each batch. The values represent
                         the durations over which the averaging needs to be done.

    Returns:
    avg (torch.Tensor): A 3D tensor of shape [B, 1, T_en] where B is the batch size,
                        T_en is the number of elements in each batch. The values represent
                        the average of the input values over the specified durations.

    Note:
    The function uses PyTorch operations for efficient computation on GPU.

    Shapes:
        - values: :math:`[B, 1, T_de]`
        - durs: :math:`[B, T_en]`
        - avg: :math:`[B, 1, T_en]`
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).float()
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).float()

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems)
    return avg
