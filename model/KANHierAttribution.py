import torch
from torch_scatter import scatter


def _fourier_delta_phi(layer, x_hat, x0_hat):
    freqs = layer.fourier_freqs
    coeffs = layer.fourier_coeffs

    def _phi(x):
        kx = x.unsqueeze(-1) * freqs
        cos = torch.cos(kx).unsqueeze(-1) * coeffs[0]
        sin = torch.sin(kx).unsqueeze(-1) * coeffs[1]
        return cos.sum(dim=-2) + sin.sum(dim=-2)

    return _phi(x_hat) - _phi(x0_hat)


def _stable_signed_edge_weights(delta_x, dst, num_dst, eps=1e-8):
    denom = scatter(delta_x.abs(), dst, dim=1, dim_size=num_dst, reduce="sum") + eps
    w_edge = delta_x / denom[:, dst, :]
    return w_edge


def propagate_relevance(
    x_src,
    x0_src,
    edge_index,
    kan_layer,
    num_dst,
    R_dst=None,
    eps=1e-8,
    reduce_edge_channel="mean",
):
    B, N_src, _ = x_src.shape
    src, dst = edge_index
    agg = kan_layer.aggregation

    x_hat = scatter(x_src[:, src], dst, dim=1, dim_size=num_dst, reduce=agg)
    x0_hat = scatter(x0_src[:, src], dst, dim=1, dim_size=num_dst, reduce=agg)
    delta_phi = _fourier_delta_phi(kan_layer, x_hat, x0_hat)
    R_dst_seed = delta_phi.sum(dim=(2, 3))
    R_dst_used = R_dst_seed if R_dst is None else R_dst
    delta_x = x_src[:, src] - x0_src[:, src]
    w_edge = _stable_signed_edge_weights(delta_x, dst, num_dst, eps=eps)
    if reduce_edge_channel == "sum":
        w_edge_scalar = w_edge.sum(dim=-1)
    else:
        w_edge_scalar = w_edge.mean(dim=-1)
    R_edge = R_dst_used[:, dst] * w_edge_scalar
    R_src = scatter(
        R_edge,
        src.unsqueeze(0).expand(B, -1),
        dim=1,
        dim_size=N_src,
        reduce="sum",
    )
    return R_src


def cascade_gene_go_ke(model_wrapper, input_t, baseline_t, edges):
    gene_go, go_ke, ke_ke = edges[0], edges[1], edges[2]
    kavnn = model_wrapper.kan

    with torch.no_grad():
        gene = kavnn.gene_layer(input_t[:, :-2048].unsqueeze(-1))
        gene0 = kavnn.gene_layer(baseline_t[:, :-2048].unsqueeze(-1))

        go = kavnn.gene2go(gene, gene_go)
        go = kavnn.go_encode_layer(go)
        go = kavnn.go_decode_layer(go.unsqueeze(-1))

        go0 = kavnn.gene2go(gene0, gene_go)
        go0 = kavnn.go_encode_layer(go0)
        go0 = kavnn.go_decode_layer(go0.unsqueeze(-1))

        ke = kavnn.go2ke(go, go_ke)
        ke0 = kavnn.go2ke(go0, go_ke)
        for layer in kavnn.ke2ke:
            ke = layer(ke, ke_ke)
            ke0 = layer(ke0, ke_ke)

    R_ke = (ke - ke0).mean(dim=2)
    R_go = propagate_relevance(
        go, go0, go_ke, kavnn.go2ke, kavnn.go2ke.num_nodes, R_dst=R_ke
    )
    R_gene = propagate_relevance(
        gene, gene0, gene_go, kavnn.gene2go, kavnn.gene2go.num_nodes, R_dst=R_go
    )
    return R_gene.abs().mean(0), R_go.abs().mean(0), R_ke.abs().mean(0)
