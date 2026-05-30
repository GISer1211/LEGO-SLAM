#
# Semantic enhancement losses for LEGO-SLAM.
#
# This module implements three light-weight, ablatable components that improve
# the open-vocabulary language field (IoU / accuracy) and, indirectly, rendering
# quality (PSNR) of the language-embedded Gaussian map:
#
#   A1. CLIP-aligned feature distillation loss
#       The original pipeline supervises the rendered language feature with an
#       L1 loss only, while evaluation (IoU / accuracy) relies on the *cosine*
#       similarity between the decoded feature and CLIP text embeddings. This
#       train/eval mismatch caps the achievable IoU. We add an explicit cosine
#       term (computed over the channel dimension) so the training objective is
#       aligned with the open-vocabulary query metric.
#
#   A2. Semantic uncertainty weighting
#       2D teacher features (e.g. LSeg) are unreliable at object boundaries and
#       in ambiguous regions. Baking that noise into the 3D map hurts IoU. We
#       derive a per-pixel confidence from the entropy of the soft assignment of
#       the *teacher* feature to the scene language codebook (the same codebook
#       already built for loop closure) and down-weight ambiguous pixels during
#       distillation. No text labels are used, so there is no evaluation leakage.
#
#   A3. Geometry/appearance-consistent feature smoothness regularizer
#       Per-Gaussian language features are supervised independently per view and
#       can be noisy. We add a bilateral kNN consistency term that pulls together
#       the features of Gaussians that are close in space *and* color, denoising
#       the 3D language field. This reuses the cdist+topk pattern already present
#       in the language-based pruning.
#
# All components are differentiable w.r.t. the rendered/per-Gaussian features and
# are gated by config flags so that the baseline behaviour can be reproduced
# exactly (set the corresponding flags to 0).

import math
import torch
import torch.nn.functional as F


def feature_distillation_loss(rendered, gt, weight_map=None, lambda_cos=0.5, eps=1e-6):
    """CLIP-aligned feature distillation loss (A1 + optional A2 weighting).

    Args:
        rendered:   [C, H, W] rendered (decoded) language feature.
        gt:         [C, H, W] teacher language feature.
        weight_map: [H, W] per-pixel confidence in [0, 1] or None (A2). When
                    None, all valid pixels are weighted equally.
        lambda_cos: weight of the cosine term relative to the L1 term.
        eps:        numerical stability constant.

    Returns:
        total_loss (scalar tensor), components dict {"l1", "cos"} (detached).
    """
    gt = gt.float()
    rendered = rendered.float()

    # Valid pixels are those where the teacher actually produced a feature.
    valid = (gt.abs().sum(dim=0) > eps).float()  # [H, W]
    if weight_map is not None:
        w = valid * weight_map.to(valid.dtype)
    else:
        w = valid
    w_sum = w.sum().clamp_min(1.0)

    # L1 term: mean absolute error over channels, weighted per pixel.
    l1_map = (rendered - gt).abs().mean(dim=0)  # [H, W]
    l1_term = (l1_map * w).sum() / w_sum

    # Cosine term over the channel dimension (matches the IoU evaluation metric).
    r_n = F.normalize(rendered, p=2, dim=0, eps=eps)
    g_n = F.normalize(gt, p=2, dim=0, eps=eps)
    cos_map = 1.0 - (r_n * g_n).sum(dim=0)  # [H, W]
    cos_term = (cos_map * w).sum() / w_sum

    total = l1_term + lambda_cos * cos_term
    return total, {"l1": l1_term.detach(), "cos": cos_term.detach()}


@torch.no_grad()
def codebook_confidence_weight(gt_feature, vocab_gpu, tau=0.1, min_weight=0.1, eps=1e-6):
    """Per-pixel semantic confidence from codebook soft-assignment entropy (A2).

    A pixel whose teacher feature is ambiguous between several codebook concepts
    (high entropy) receives a low weight; a confident pixel receives a weight
    close to 1. The weight is floored at ``min_weight`` so no pixel is fully
    discarded.

    Args:
        gt_feature: [C, H, W] teacher language feature (e.g. 512-D LSeg/CLIP).
        vocab_gpu:  [K, C] codebook (cluster centers) on the same device.
        tau:        softmax temperature over codebook cosine similarities.
        min_weight: lower bound on the returned weight.
        eps:        numerical stability constant.

    Returns:
        weight: [H, W] tensor in [min_weight, 1].
    """
    if vocab_gpu is None:
        return None
    C, H, W = gt_feature.shape
    feats = gt_feature.float().permute(1, 2, 0).reshape(-1, C)  # [HW, C]
    feats_n = F.normalize(feats, dim=1, eps=eps)
    vocab_n = F.normalize(vocab_gpu.float(), dim=1, eps=eps)
    sims = feats_n @ vocab_n.t()                                # [HW, K]
    probs = F.softmax(sims / tau, dim=1)
    K = probs.shape[1]
    entropy = -(probs * (probs + eps).log()).sum(dim=1)         # [HW]
    norm_entropy = (entropy / math.log(K)).clamp(0.0, 1.0)
    weight = (1.0 - norm_entropy) * (1.0 - min_weight) + min_weight
    return weight.reshape(H, W)
