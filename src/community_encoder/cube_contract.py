"""Feature layout and kernel semantics shared by cube producers and consumers."""
import copy
import hashlib
from pathlib import Path


def artifact_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"name": path.name, "size": path.stat().st_size, "content_sha256": h.hexdigest()}


def add_basis_identities(contract):
    """Pin each block's actual basis, not only its mutable directory name."""
    import numpy as np
    for b in contract["feature_blocks"]:
        basis = Path(b["esk_basis_dir"])
        b["basis_identity"] = {n: artifact_identity(basis / n) for n in
                               ("meta.json", "esk_projmat.npy", "esk_landmarks.npy")}
        if b["start"]:
            target = Path(b["coordinate_target"])
            with np.load(target) as coords:
                if coords["Z"].shape[-1] != b["stop"] - b["start"]:
                    raise ValueError("coordinate-target width differs from exported block")
            b["coordinate_target_identity"] = artifact_identity(target)
    return contract


def export_contract(latent_dim, basis_dir, private_dim=0, heads=()):
    """Describe the exported coordinates; never label arbitrary private heads as ESK."""
    blocks = [{"name": "summer", "start": 0, "stop": int(latent_dim),
               "kernel": "ruzicka", "centered": False,
               "esk_basis_dir": str(basis_dir), "coordinate_target": "bbs_esk"}]
    offset = int(latent_dim)
    for head in heads if private_dim else ():
        width = int(head.get("private", 0))
        if not width:
            continue
        coord = head.get("coord_target") or {}
        if not (head.get("private_only") and head.get("identity")
                and coord.get("basis_dir") and coord.get("path")):
            raise ValueError("exported private blocks require identity/private_only heads "
                             "and coord_target.path/basis_dir provenance")
        blocks.append({"name": str(head["label"]), "start": offset, "stop": offset + width,
                       "kernel": "ruzicka", "centered": False,
                       "esk_basis_dir": str(coord["basis_dir"]),
                       "coordinate_target": str(coord["path"])})
        offset += width
    if offset != int(latent_dim) + int(private_dim):
        raise ValueError("private head widths do not match checkpoint private_dim")
    result = {"schema_version": 2, "kernel": "ruzicka_sum" if private_dim else "ruzicka",
              "centered": False, "latent_dim": offset, "feature_blocks": blocks,
              "kernel_contract": ("Z(x) dot Z(x') approximates the sum of the uncentered "
                                  "Ruzicka kernels in feature_blocks" if private_dim else
                                  "Z(x) dot Z(x') ~= uncentered Ruzicka(x,x')")}
    validate_cube_contract(result)
    return result


def validate_cube_contract(contract, width=None):
    """Accept legacy single-basis cubes, require an explicit contiguous sum layout."""
    kernel = contract.get("kernel")
    if kernel not in ("ruzicka", "ruzicka_sum") or contract.get("centered") is not False:
        raise ValueError("cube requires uncentered ruzicka or ruzicka_sum features")
    size = int(contract.get("latent_dim", 0))
    if size <= 0 or (width is not None and size != int(width)):
        raise ValueError(f"cube contract latent_dim={size} != array width {width}")
    blocks = contract.get("feature_blocks")
    if kernel == "ruzicka_sum" and (not blocks or len(blocks) < 2):
        raise ValueError("sum kernel requires at least two explicit feature_blocks")
    if blocks:
        offset, names = 0, set()
        for b in blocks:
            if (b.get("start") != offset or int(b.get("stop", 0)) <= offset
                    or b.get("kernel") != "ruzicka" or b.get("centered") is not False
                    or not b.get("esk_basis_dir") or not b.get("name")
                    or b["name"] in names):
                raise ValueError("invalid, overlapping, or unidentified cube feature block")
            names.add(b["name"])
            offset = int(b["stop"])
        if offset != size or (kernel == "ruzicka" and len(blocks) != 1):
            raise ValueError("feature_blocks do not match cube kernel/latent_dim")
    return contract


def compatible_contract(contract):
    """Semantic fields that must agree across ensemble members (not checkpoint IDs)."""
    validate_cube_contract(contract)
    keys = ("schema_version", "kernel", "centered", "latent_dim", "feature_blocks",
            "temporal_output", "esk_basis_dir")
    return {k: copy.deepcopy(contract[k]) for k in keys if k in contract}


def select_block_prefixes(contract, columns):
    """Explicit within-season eigenfeature selection; preserve both seasons and provenance."""
    validate_cube_contract(contract)
    columns = list(columns)
    if not columns or any(type(i) is not int for i in columns) or columns != sorted(set(columns)):
        raise ValueError("feature_columns must be nonempty, sorted unique integer indices")
    width = int(contract["latent_dim"])
    if columns[0] < 0 or columns[-1] >= width:
        raise ValueError("feature_columns exceed source width")
    blocks = contract.get("feature_blocks")
    result = copy.deepcopy(contract)
    if contract["kernel"] == "ruzicka_sum" and not blocks:
        raise ValueError("sum feature selection requires source blocks")
    retained, offset = [], 0
    if blocks:
        for b in blocks:
            chosen = [i for i in columns if b["start"] <= i < b["stop"]]
            if not chosen or chosen != list(range(b["start"], b["start"] + len(chosen))):
                raise ValueError("select a nonempty leading prefix of every seasonal block")
            retained.append({**b, "start": offset, "stop": offset + len(chosen)})
            offset += len(chosen)
        result["source_feature_blocks"] = copy.deepcopy(blocks)
        result["feature_blocks"] = retained
    elif columns != list(range(len(columns))):
        raise ValueError("legacy single-basis features require a leading prefix")
    result.update(latent_dim=len(columns), source_latent_dim=width, feature_columns=columns,
                  truncation=("none" if len(columns) == width else
                              "per_block_top_eigenfeatures" if contract["kernel"] == "ruzicka_sum" else "top_eigenfeatures"))
    validate_cube_contract(result)
    return result


def validate_model_feature_contract(contract, width):
    """Equal isotropic coefficient variance gives the explicitly declared seasonal sum kernel."""
    validate_cube_contract(contract, width)
    if contract.get("feature_prior") != "isotropic":
        raise ValueError("environment kernel requires isotropic feature coefficients")
    source = int(contract.get("source_latent_dim", width))
    if source < width:
        raise ValueError("source width is smaller than retained width")
    if source > width:
        truncation = contract.get("truncation")
        if contract["kernel"] == "ruzicka_sum":
            if truncation != "per_block_top_eigenfeatures":
                raise ValueError("sum kernel cannot use a positional global eigenfeature cut")
            original = {**contract, "latent_dim": source,
                        "feature_blocks": contract.get("source_feature_blocks")}
            selected = select_block_prefixes(original, contract.get("feature_columns", []))
            if selected["feature_blocks"] != contract["feature_blocks"] or selected["latent_dim"] != width:
                raise ValueError("retained seasonal blocks disagree with source selection")
        elif truncation != "top_eigenfeatures":
            raise ValueError("invalid single-basis truncation")
    return contract
