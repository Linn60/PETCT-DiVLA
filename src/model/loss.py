import torch
import torch.nn.functional as F
import torch.distributed as dist


def _all_reduce_scalar(t):
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t)
    return t


def constrastive_loss(image_mu, text_mu, logit_scale):
    logits_per_image = logit_scale * image_mu @ text_mu.T
    logits_per_text = logits_per_image.T

    n = logits_per_image.shape[0]
    targets = torch.arange(n, device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    loss = (loss_i2t + loss_t2i) / 2
    return loss


def global_alignment_loss(t_global, v_fus, logit_scale, logit_bias):
    sim = logit_scale * (t_global @ v_fus.T) + logit_bias
    B = sim.shape[0]
    labels = 2 * torch.eye(B, device=sim.device) - 1
    return -F.logsigmoid(labels * sim).mean()


def local_alignment_loss(t_pet, t_ct, t_shr,
                         v_pet_o, v_ct_o, v_fus_o,
                         sent_patient_idx, sent_organ_idx,
                         organ_labels, logit_scale, logit_bias):
    S = t_pet.shape[0]
    if S == 0:
        return torch.tensor(0.0, device=t_pet.device)

    dev = t_pet.device
    n_vec = sent_patient_idx.to(dev)
    o_vec = sent_organ_idx.to(dev)
    labels = organ_labels.to(dev)

    same_organ = (o_vec.unsqueeze(1) == o_vec.unsqueeze(0))
    same_patient = (n_vec.unsqueeze(1) == n_vec.unsqueeze(0))
    y_self = labels[n_vec, o_vec]
    both_neg = (y_self.unsqueeze(1) == 0) & (y_self.unsqueeze(0) == 0)

    exclude_mask = same_organ & ~same_patient & both_neg

    pos_mask = same_organ & same_patient
    sig_labels = (2 * pos_mask.float() - 1).unsqueeze(0)

    t_stack = torch.stack([t_pet, t_ct, t_shr])
    v_stack = torch.stack([v_pet_o, v_ct_o, v_fus_o])
    sim = logit_scale * torch.bmm(t_stack, v_stack.transpose(1, 2)) + logit_bias

    loss_matrix = -F.logsigmoid(sig_labels * sim)
    compute_mask = (~exclude_mask).unsqueeze(0).float()
    return (loss_matrix * compute_mask).sum() / (compute_mask.sum() * loss_matrix.shape[0]).clamp(min=1)


def local_alignment_loss_fus_only(t_shr, v_fus_o,
                                  sent_patient_idx, sent_organ_idx,
                                  organ_labels, logit_scale, logit_bias):
    S = t_shr.shape[0]
    if S == 0:
        return torch.tensor(0.0, device=t_shr.device)

    dev = t_shr.device
    n_vec = sent_patient_idx.to(dev)
    o_vec = sent_organ_idx.to(dev)
    labels = organ_labels.to(dev)

    same_organ = (o_vec.unsqueeze(1) == o_vec.unsqueeze(0))
    same_patient = (n_vec.unsqueeze(1) == n_vec.unsqueeze(0))
    y_self = labels[n_vec, o_vec]
    both_neg = (y_self.unsqueeze(1) == 0) & (y_self.unsqueeze(0) == 0)

    exclude_mask = same_organ & ~same_patient & both_neg

    pos_mask = same_organ & same_patient
    sig_labels = 2 * pos_mask.float() - 1

    sim = logit_scale * (t_shr @ v_fus_o.T) + logit_bias

    loss_matrix = -F.logsigmoid(sig_labels * sim)
    compute_mask = ~exclude_mask
    return (loss_matrix * compute_mask.float()).sum() / compute_mask.float().sum().clamp(min=1)


def exclusivity_loss(t_pet, t_ct, v_pet_o, v_ct_o, alpha_pet, alpha_ct, margin=0.1):
    sim_pet_pet = (t_pet * v_pet_o).sum(dim=-1)
    sim_pet_ct = (t_pet * v_ct_o).sum(dim=-1)
    sim_ct_ct = (t_ct * v_ct_o).sum(dim=-1)
    sim_ct_pet = (t_ct * v_pet_o).sum(dim=-1)

    loss_pet = alpha_pet * torch.relu(margin + sim_pet_ct - sim_pet_pet)
    loss_ct = alpha_ct * torch.relu(margin + sim_ct_pet - sim_ct_ct)
    return (loss_pet + loss_ct).mean()


def conjunction_loss(t_shr, v_pet_o, v_ct_o, alpha_shr):
    u_pet = (t_shr * v_pet_o).sum(dim=-1)
    u_ct = (t_shr * v_ct_o).sum(dim=-1)
    u_and = -torch.logaddexp(-u_pet, -u_ct)
    return (alpha_shr * (-u_and)).mean()


def router_entropy_loss(w, attention_mask):
    mask = attention_mask.float()
    ent = -(w * torch.log(w + 1e-8)).sum(dim=-1)
    num = _all_reduce_scalar((ent * mask).sum())
    den = _all_reduce_scalar(mask.sum()).clamp(min=1)
    return num / den


def router_prior_loss(w, attention_mask, pi_0):
    mask = attention_mask.unsqueeze(-1).float()
    weighted_sum = _all_reduce_scalar((w * mask).sum(dim=(0, 1)))
    total_tokens = _all_reduce_scalar(mask.sum()).clamp(min=1)
    pi = weighted_sum / total_tokens
    pi_0_t = torch.tensor(pi_0, device=w.device, dtype=w.dtype)
    return F.kl_div(torch.log(pi + 1e-8), pi_0_t, reduction='sum')


def _sim_stats(sim: torch.Tensor) -> dict:
    """
    Log-friendly statistics for similarity matrix.
    - diag: positives (matched pairs)
    - offdiag: negatives (mismatched pairs)
    """
    # NOTE: this is purely for logging; keep it out of autograd.
    with torch.no_grad():
        sim_f = sim.detach().float()
        dev = sim_f.device
        B = sim_f.shape[0]
        eye = torch.eye(B, device=dev, dtype=torch.bool)
        diag = sim_f.diagonal()
        off = sim_f[~eye]

        def _nan_scalar() -> torch.Tensor:
            return torch.tensor(float("nan"), device=dev, dtype=torch.float32)

        def _q(x, p: float) -> torch.Tensor:
            if x.numel() == 0:
                return _nan_scalar()
            return torch.quantile(x, torch.tensor(p, device=dev, dtype=torch.float32))

        def _mean(x) -> torch.Tensor:
            if x.numel() == 0:
                return _nan_scalar()
            return x.mean()

        # Return 0-dim tensors so upstream logging can safely .detach()
        return {
            "sim_diag_mean": _mean(diag),
            "sim_diag_p10": _q(diag, 0.10),
            "sim_diag_p50": _q(diag, 0.50),
            "sim_diag_p90": _q(diag, 0.90),
            "sim_off_mean": _mean(off),
            "sim_off_p10": _q(off, 0.10),
            "sim_off_p50": _q(off, 0.50),
            "sim_off_p90": _q(off, 0.90),
        }


def _embed_stats(t_global: torch.Tensor, v_fus: torch.Tensor,
                  logit_scale: torch.Tensor) -> dict:
    """
    Intra-modal pairwise cosine similarity (mean & std) + logit_scale.
    If one modality collapses to a single vector, its intra-modal cosine → 1.
    """
    with torch.no_grad():
        t = t_global.detach().float()
        v = v_fus.detach().float()

        def _intra(x):
            sim = x @ x.T
            B = sim.shape[0]
            mask = ~torch.eye(B, device=sim.device, dtype=torch.bool)
            off = sim[mask]
            return off.mean(), off.std()

        t_mean, t_std = _intra(t)
        v_mean, v_std = _intra(v)

        return {
            "txt_intra_cos_mean": t_mean,
            "txt_intra_cos_std": t_std,
            "vis_intra_cos_mean": v_mean,
            "vis_intra_cos_std": v_std,
            "logit_scale": logit_scale.detach().float(),
        }


def petctclip_loss(features, config):
    t_global = features["t_global"]
    v_fus = features["v_fus"]
    logit_scale = features["logit_scale"]
    logit_bias = features["logit_bias"]

    L_global = global_alignment_loss(t_global, v_fus, logit_scale, logit_bias)
    losses = {"L_global": L_global}

    # Similarity matrix stats for debugging retrieval learning dynamics.
    sim = logit_scale * (t_global @ v_fus.T)
    losses.update(_sim_stats(sim))
    losses.update(_embed_stats(t_global, v_fus, logit_scale))
    total = L_global

    if "t_pet" in features and "v_pet_o" in features:
        L_local = local_alignment_loss(
            features["t_pet"], features["t_ct"], features["t_shr"],
            features["v_pet_o"], features["v_ct_o"], features["v_fus_o"],
            features["sent_patient_idx"], features["sent_organ_idx"],
            features["organ_labels"], logit_scale, logit_bias,
        )
        losses["L_local"] = L_local
        total = total + config.lambda_loc * L_local

    if "w" in features:
        sent_attention_mask = features.get("sent_attention_mask")
        if sent_attention_mask is None:
            S, L, _ = features["w"].shape
            sent_attention_mask = torch.ones(S, L, device=features["w"].device)
        L_ent = router_entropy_loss(features["w"], sent_attention_mask)
        L_prior = router_prior_loss(features["w"], sent_attention_mask, config.pi_0)
        losses["L_ent"] = L_ent
        losses["L_prior"] = L_prior
        total = total + config.lambda_ent * L_ent + config.lambda_pr * L_prior

    losses["total"] = total
    return total, losses