"""C09: binary64 single-query, single-head attention reference (stdlib only).

Batch/head dimensions can be handled by an outer loop. Scores include scaling,
bias and masks before regional decomposition. This is an audit reference,
not a kernel or a trained router. See REPORT.md for assumptions and costs.
"""
import math
import sys

MAX_TOKENS = 1024
NEG_INF = float("-inf")


def _scores(xs):
    xs = [float(x) for x in xs]
    if any(math.isnan(x) or x == math.inf for x in xs):
        raise ValueError("NaN/+inf logits are undefined here; -inf masks are allowed")
    return xs


def logsumexp(xs):
    """Stable LSE; empty/all-masked input -> -inf. Reject NaN and +inf."""
    xs = _scores(xs)
    if not xs or max(xs) == NEG_INF:
        return NEG_INF
    m = max(xs)
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def softmax(xs):
    """Stable shifted exponentials; no subtracting a rounded absolute LSE."""
    xs = _scores(xs)
    if not xs or max(xs) == NEG_INF:
        raise ValueError("attention needs at least one unmasked token")
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    z = math.fsum(e)
    return [x / z for x in e]


def partition(regions, n):
    """Exact disjoint cover, permitting empty regions. Reject overlaps/duplicates."""
    if not 1 <= n <= MAX_TOKENS:
        raise ValueError("token count must be 1..1024")
    regions = [list(r) for r in regions]
    ids = [i for r in regions for i in r]
    if any(type(i) is not int for i in ids):
        raise ValueError("indices must be integers")
    if sorted(ids) != list(range(n)):
        raise ValueError("regions must partition tokens exactly once")
    return regions


def _inputs(logits, values):
    s = _scores(logits)
    if not 1 <= len(s) <= MAX_TOKENS or len(s) != len(values):
        raise ValueError("token/value count mismatch or outside 1..1024")
    v = [[float(x) for x in row] for row in values]
    if not v or not v[0] or any(len(x) != len(v[0]) for x in v):
        raise ValueError("values must be rectangular and nonempty")
    if any(not math.isfinite(x) for row in v for x in row):
        raise ValueError("values must be finite")
    return s, v


def weighted_sum(weights, values):
    return [math.fsum(w * v[j] for w, v in zip(weights, values))
            for j in range(len(values[0]))]


def dense_attention(logits, values):
    s, v = _inputs(logits, values)
    p = softmax(s)
    return {"output": weighted_sum(p, v), "probabilities": p}


def attention_logits(query, keys):
    """q dot k / sqrt(d); no rotary/bias assumption is imposed elsewhere."""
    q = [float(x) for x in query]
    if not q or not keys or len(keys) > MAX_TOKENS:
        raise ValueError("invalid q/K shape")
    if any(not math.isfinite(x) for x in q):
        raise ValueError("query must be finite")
    out = []
    for k in keys:
        if len(k) != len(q) or any(not math.isfinite(x) for x in k):
            raise ValueError("keys must be finite and match q dimension")
        # Scale q first to reduce avoidable multiplication overflow.
        s = math.fsum((x / math.sqrt(len(q))) * float(y)
                      for x, y in zip(q, k))
        if not math.isfinite(s):
            raise ValueError("dot product overflow; rescale input")
        out.append(s)
    return out


def hierarchical_attention(logits, values, regions):
    s, v = _inputs(logits, values)
    regions = partition(regions, len(s))
    if max(s) == NEG_INF:
        raise ValueError("all tokens masked")
    # All regional masses use the SAME global shift. Computing absolute
    # LSEs first loses log(n) when the shared offset is e.g. 1e300.
    shift = max(s)
    shifted = [x - shift for x in s]
    region_log_mass = [logsumexp([shifted[i] for i in r]) for r in regions]
    region_weights = softmax(region_log_mass)
    conditionals, region_outputs = [], []
    for r, mass in zip(regions, region_log_mass):
        if mass == NEG_INF:
            conditionals.append([0.0] * len(r))
            region_outputs.append([0.0] * len(v[0]))
        else:
            a = softmax([s[i] for i in r])
            conditionals.append(a)
            region_outputs.append(weighted_sum(a, [v[i] for i in r]))
    return {"output": weighted_sum(region_weights, region_outputs),
            "region_probabilities": region_weights,
            "conditional_probabilities": conditionals,
            "region_outputs": region_outputs,
            "region_log_mass_shifted": region_log_mass,
            "common_shift": shift}


def approximate_region_logits(logits, regions, mode):
    """Diagnostic aggregation of supplied fine logits (therefore full fine read).

    mean_key equals mean score only for a linear dot product and consistently
    averaged additive bias. For real cheap routing use precomputed centroids;
    this function does NOT implement a cost-free max/LSE oracle.
    """
    s = _scores(logits)
    regions = partition(regions, len(s))
    if mode not in ("mean_key", "max", "logsumexp", "mean_log_size"):
        raise ValueError("unknown approximation")
    if max(s) == NEG_INF:
        raise ValueError("all tokens masked")
    shift = max(s)
    out = []
    for r in regions:
        xs = [s[i] - shift for i in r if s[i] != NEG_INF]
        if not xs:
            out.append(NEG_INF)
        elif mode == "logsumexp":
            out.append(logsumexp(xs))
        elif mode == "max":
            out.append(max(xs))
        else:
            # Mask policy: count/mean only eligible tokens. Coarse centroids
            # cannot in general support arbitrary query-dependent fine masks.
            mean = math.fsum(x / len(xs) for x in xs)
            out.append(mean + (math.log(len(xs)) if mode == "mean_log_size" else 0))
    return out


def select_topk(region_scores, k):
    scores = _scores(region_scores)
    eligible = [i for i, s in enumerate(scores) if s != NEG_INF]
    if type(k) is not int or not 1 <= k <= len(eligible):
        raise ValueError("k must be 1..number of nonempty/unmasked regions")
    return sorted(eligible, key=lambda i: (-scores[i], i))[:k]


def selected_attention(logits, values, regions, selected):
    """Drop excluded regions and exactly renormalize the selected fine tokens."""
    s, v = _inputs(logits, values)
    regions = partition(regions, len(s))
    selected = list(selected)
    if not selected or any(type(i) is not int or i < 0 or i >= len(regions)
                           for i in selected) or len(set(selected)) != len(selected):
        raise ValueError("selected region ids must be unique and in range")
    ids = [i for r in selected for i in regions[r]]
    if not ids:
        raise ValueError("selection contains no tokens")
    out = dense_attention([s[i] for i in ids], [v[i] for i in ids])
    return {"output": out["output"], "indices": ids,
            "conditional_probabilities": out["probabilities"]}


def l2(x):
    return math.hypot(*x)


def truncation_diagnostic(logits, values, regions, selected):
    """Full-read diagnostic, never a cheap routing implementation.

    If ||v_i||_2 <= B and p is original dense softmax, exact conditional
    selection satisfies ||y-y_S|| <= 2 B epsilon, epsilon = sum dropped p_i.
    Directly sum dropped mass (avoid 1 - retained cancellation).
    """
    s, v = _inputs(logits, values)
    full = dense_attention(s, v)
    sub = selected_attention(s, v, regions, selected)
    ids = set(sub["indices"])
    retained = math.fsum(p for i, p in enumerate(full["probabilities"]) if i in ids)
    omitted = math.fsum(p for i, p in enumerate(full["probabilities"]) if i not in ids)
    b = max(l2(x) for x in v)
    error = l2([x-y for x, y in zip(full["output"], sub["output"])])
    return {"retained_mass": retained, "omitted_mass": omitted,
            "l2_output_error": error, "value_norm_bound": b,
            "error_bound_2B_epsilon": 2*b*omitted,
            "selected_fine_tokens": len(ids),
            "diagnostic_reads_all_keys_and_values": True}


def binary64_environment():
    return {"mant_dig": sys.float_info.mant_dig,
            "max_exp": sys.float_info.max_exp,
            "reference_dtype": "IEEE754 binary64 (Python float)",
            "maximum_fine_tokens": MAX_TOKENS}


def prepare_centroids(keys, regions):
    """Offline/static unmasked K only; costs N*d fine-key reads/additions.

    Rebuild when K/partition changes. Arbitrary query-dependent masks and
    position biases require additional information not represented here.
    """
    if not keys or not keys[0] or any(len(k) != len(keys[0]) for k in keys):
        raise ValueError("invalid key matrix")
    regions = partition(regions, len(keys))
    return [([math.fsum(float(keys[i][j])/len(r) for i in r)
              for j in range(len(keys[0]))] if r else None) for r in regions]


def route_attention(query, keys, values, regions, k, mode, centroids=None):
    """Actual small online route experiment; no precomputed fine logits.

    Centroid routes read only chosen fine keys online; exact max/LSE routes
    read ALL fine keys online. Diagnostic dense evaluation is separate.
    No mask or additive position bias is supported by this convenience API.
    """
    regions = partition(regions, len(keys))
    if len(values) != len(keys):
        raise ValueError("K/V length mismatch")
    if mode not in ("mean_key", "mean_log_size", "max", "logsumexp"):
        raise ValueError("unknown router")
    r_count = sum(bool(r) for r in regions)
    if mode in ("mean_key", "mean_log_size"):
        if centroids is None or len(centroids) != len(regions):
            raise ValueError("supply offline centroids consistent with K/regions")
        router = []
        for r, c in zip(regions, centroids):
            if not r:
                router.append(NEG_INF)
            elif c is None:
                raise ValueError("missing nonempty centroid")
            else:
                score = attention_logits(query, [c])[0]
                router.append(score + (math.log(len(r))
                                        if mode == "mean_log_size" else 0.))
        selected = select_topk(router, k)
        ids = [i for r in selected for i in regions[r]]
        fine = attention_logits(query, [keys[i] for i in ids])
        fine_qk = len(ids)
        coarse_qk = r_count
        router_exp = 0
    else:
        all_scores = attention_logits(query, keys)
        router = approximate_region_logits(all_scores, regions, mode)
        selected = select_topk(router, k)
        ids = [i for r in selected for i in regions[r]]
        fine = [all_scores[i] for i in ids]  # reuse, do not score twice
        fine_qk = len(keys)
        coarse_qk = 0
        router_exp = len(keys) if mode == "logsumexp" else 0
    out = dense_attention(fine, [values[i] for i in ids])["output"]
    return {"output": out, "selected_regions": selected,
            "selected_fine_tokens": len(ids),
            "cost": {
                "online_fine_key_rows_read": fine_qk,
                "online_centroid_rows_read": coarse_qk,
                "online_fine_value_rows_read": len(ids),
                "qk_scalar_multiply_add_pairs": (fine_qk+coarse_qk)*len(query),
                "av_scalar_multiply_add_pairs": len(ids)*len(values[0]),
                "exp_evaluations": router_exp+len(ids),
                "region_rank_items": r_count,
                "fine_key_scans_for_max_or_lse": fine_qk if coarse_qk == 0 else 0,
                "cost_is_analytical_counts_not_timing": True}}
