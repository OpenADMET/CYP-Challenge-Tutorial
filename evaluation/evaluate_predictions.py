"""Functions for evaluating the predictions of the OpenADMET CYP blind challenge."""

import inspect
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from .config import (
    ACTIVITY_METRICS,
    BISYRMSD_NAN_PENALTY,
    BOOTSTRAP_SAMPLES,
    CLASSIFICATION_ENDPOINTS,
    CLASSIFICATION_METRICS,
    ENDPOINTS_TO_LOG_TRANSFORM,
    MACRO_ENDPOINT_LABEL,
    METRIC_NAN_FALLBACK,
    POSEBUSTERS_MAX_FAILURES,
    REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX,
    REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX,
    STRUCTURE_METRICS,
)
from .utils import bootstrap_sampling, clip_and_log_transform

# Residue names treated as solvent — excluded from ligand detection
_SOLVENT_RESIDUE_NAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD"})

# Sentinel returned when a per-compound structure score cannot be computed
_NAN_STRUCTURE_METRICS: dict[str, float] = {m: np.nan for m in STRUCTURE_METRICS}


def _metrics_for_endpoint(endpoint: str) -> list[tuple[str, Callable]]:
    """Return the metric list to use for a given activity endpoint.

    Classification (TDI) endpoints are scored with ``CLASSIFICATION_METRICS``
    (MCC/Accuracy/Precision/Recall/F1); every other activity endpoint (regression,
    direct-inhibition pIC50) is scored with ``ACTIVITY_METRICS``.
    """
    return (
        CLASSIFICATION_METRICS
        if endpoint in CLASSIFICATION_ENDPOINTS
        else ACTIVITY_METRICS
    )


def score_activity_predictions(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, endpoints: list[str]
) -> pd.DataFrame:
    """Score the activity predictions against the ground truth.

    Metrics are calculated for bootstrapped samples of the dataset to allow for testing
    the statistical significance of differences between submissions. Each endpoint is
    scored with the metric list appropriate to its type — regression endpoints get
    ``ACTIVITY_METRICS``, classification (TDI) endpoints get ``CLASSIFICATION_METRICS``
    — see ``_metrics_for_endpoint``.

    Each endpoint is scored only on the compounds that have a ground-truth value for
    that endpoint — a compound not tested for a given endpoint has ``y_true == NaN``
    there and is excluded from that endpoint's bootstrap sampling entirely. Every
    compound with a ground-truth value is expected to have a prediction (participants
    are asked to predict every compound, and submission_validation.py is the primary
    check for missing predictions); a NaN prediction for such a compound is treated as
    a validation failure here too.

    Regression endpoints carry credible-interval bound columns in ``ground_truth``
    (named ``f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}"`` /
    ``f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}"``), used by the
    soft-thresholded RAE metric (``ST-RAE``). When present, these are threaded through
    to ``bootstrap_metrics`` alongside ``y_true``/``y_pred``; classification endpoints
    have no such columns, so ``None`` is passed instead (harmless, since none of
    ``CLASSIFICATION_METRICS`` consume them).

    This function does not compute the macro-averaged "MA" pseudo-endpoint. Callers
    that want a track's "MA" row should call ``add_macro_endpoint`` on this function's
    output with the same ``endpoints`` (and that track's own metrics).

    Args:
        predictions (pd.DataFrame): The predicted activity values.
        ground_truth (pd.DataFrame): The true activity values.
        endpoints (list[str]): The endpoints to score, e.g. ``REGRESSION_ENDPOINTS``
            or ``CLASSIFICATION_ENDPOINTS`` — regression and classification are
            independent submission tracks, so a given call only ever scores one
            track's endpoints.

    Returns:
        pd.DataFrame: A DataFrame containing the scored bootstrapped activity
            predictions, one row per (endpoint, bootstrap sample) — no macro
            pseudo-endpoint included.

    Raises:
        ValueError: If a compound with a ground-truth value for an endpoint has no
            prediction for that endpoint.

    """
    logger.info("Scoring activity predictions against ground truth")
    merged_df = predictions.merge(
        ground_truth, on="Molecule_Name", suffixes=("_pred", "_true"), how="right"
    ).sort_values("Molecule_Name")
    logger.info(
        "Completed merging predictions with ground truth. Merged dataset contains {} "
        "rows and {} columns.",
        merged_df.shape[0],
        merged_df.shape[1],
    )

    all_endpoint_bootstrap_results_list = []
    for endpoint in endpoints:
        logger.info("Scoring endpoint: {}", endpoint)
        y_pred = merged_df[f"{endpoint}_pred"].to_numpy()
        y_true = merged_df[f"{endpoint}_true"].to_numpy()

        # Credible-interval bound columns aren't merge-suffixed: they only ever come
        # from ground_truth (predictions never carry them), so they keep their plain
        # names — see merge() above (suffixes only apply to overlapping columns).
        upper_col = f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}"
        lower_col = f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}"
        y_true_upper = (
            merged_df[upper_col].to_numpy() if upper_col in merged_df.columns else None
        )
        y_true_lower = (
            merged_df[lower_col].to_numpy() if lower_col in merged_df.columns else None
        )

        # pd.isna (not np.isnan) so this works for classification endpoints too —
        # their ground-truth column can be object/bool dtype (e.g. CYP2D6_is_TDI has
        # a couple of genuinely missing labels), which np.isnan can't handle.
        has_ground_truth = ~pd.isna(y_true)
        if not has_ground_truth.all():
            logger.debug(
                "Excluding {} compound(s) with no ground truth for endpoint {}",
                (~has_ground_truth).sum(),
                endpoint,
            )
            y_pred = y_pred[has_ground_truth]
            y_true = y_true[has_ground_truth]
            if y_true_upper is not None:
                y_true_upper = y_true_upper[has_ground_truth]
            if y_true_lower is not None:
                y_true_lower = y_true_lower[has_ground_truth]

        # A submission itself must never contain NaN (submission_validation.py's
        # nullable=False already rejects that) — this instead defends against a
        # prediction going missing specifically for a compound that *does* have
        # ground truth, which validation of the raw submission can't catch on its own.
        if pd.isna(y_pred).any():
            raise ValueError(
                f"Missing prediction(s) for endpoint '{endpoint}': every compound "
                "with a ground-truth value must have a prediction."
            )

        if endpoint in CLASSIFICATION_ENDPOINTS:
            # Safe only after NaN rows have already been dropped from both arrays.
            y_true = y_true.astype(bool)
            y_pred = y_pred.astype(bool)
        elif endpoint in ENDPOINTS_TO_LOG_TRANSFORM:
            logger.debug("Applying log transformation to endpoint {}", endpoint)
            y_pred = clip_and_log_transform(y_pred)
            y_true = clip_and_log_transform(y_true)

        bootstrap_df = bootstrap_metrics(
            y_pred,
            y_true,
            endpoint,
            n_bootstrap_samples=BOOTSTRAP_SAMPLES,
            metrics=_metrics_for_endpoint(endpoint),
            y_true_upper=y_true_upper,
            y_true_lower=y_true_lower,
        )
        all_endpoint_bootstrap_results_list.append(bootstrap_df)
    all_endpoint_bootstrap_results = pd.concat(
        all_endpoint_bootstrap_results_list, ignore_index=True
    )
    logger.info("Completed scoring activity predictions")
    return all_endpoint_bootstrap_results


def add_macro_endpoint(
    all_endpoint_bootstrap_results: pd.DataFrame,
    endpoints: list[str],
    metrics: list[tuple[str, Callable]],
) -> pd.DataFrame:
    """Narrow to one track's endpoints/metrics and append its macro "MA" row.

    ``score_activity_predictions`` scores every activity endpoint (regression and
    classification) in one call, concatenating per-endpoint frames that have
    *different* metric columns (regression rows have MAE/ST-RAE/..., classification rows
    have MCC/Accuracy/...) — the concatenated result has both sets of columns, NaN
    wherever a metric doesn't apply to that row's endpoint. This filters rows down to
    just ``endpoints`` (one track's real endpoints) and columns down to just
    ``metrics`` (that track's own metrics), so no cross-track NaN columns leak into
    the result, then appends a macro-averaged "MA" row set (via
    ``compute_macro_bootstrap_results``) when there's more than one endpoint to
    average across.

    Args:
        all_endpoint_bootstrap_results (pd.DataFrame): Output of
            ``score_activity_predictions`` (or any frame with "Sample", "Endpoint",
            and metric columns for multiple endpoints/tracks).
        endpoints (list[str]): The track's real endpoints to keep, e.g.
            ``REGRESSION_ENDPOINTS`` or ``CLASSIFICATION_ENDPOINTS``. May be empty, in
            which case the result is empty (callers should generally avoid calling
            this with an empty list rather than relying on that).
        metrics (list[tuple[str, Callable]]): The track's own metric list, e.g.
            ``ACTIVITY_METRICS`` or ``CLASSIFICATION_METRICS`` — only these columns
            are kept.

    Returns:
        pd.DataFrame: This track's real-endpoint rows, narrowed to its own metric
            columns, plus a macro "MA" row set when ``len(endpoints) > 1``.

    """
    metric_names = [name for name, _ in metrics]
    track_results = all_endpoint_bootstrap_results[
        all_endpoint_bootstrap_results["Endpoint"].isin(endpoints)
    ][["Sample", "Endpoint", *metric_names]]

    if len(endpoints) > 1:
        logger.info(
            "Calculating macro-averaged metrics across endpoints for each bootstrap sample"
        )
        macro_bootstrap_results = compute_macro_bootstrap_results(
            track_results, metrics=metrics
        )
        track_results = pd.concat(
            [track_results, macro_bootstrap_results], ignore_index=True
        )
    return track_results


def compute_macro_bootstrap_results(
    all_endpoint_bootstrap_results: pd.DataFrame,
    metrics: list[tuple[str, Callable]],
) -> pd.DataFrame:
    """Compute per-bootstrap-sample macro-averaged metrics across all endpoints.

    For every bootstrap sample, every metric in ``metrics`` is macro-averaged across
    endpoints with a plain arithmetic mean.

    Spearman_R was previously averaged via a Fisher z-transform (``arctanh`` /
    ``tanh``), the standard variance-stabilising treatment for combining several
    noisy *estimates of the same underlying correlation* (e.g. meta-analysis, or
    averaging one endpoint's Spearman across repeated resamples of the same data).
    That doesn't apply here: this average combines Spearman scores from *different*
    endpoints (different isoforms), which are unrelated true correlations, not
    repeated estimates of one. Fisher's z blows up near +/-1 (``arctanh(1) = inf``,
    clipped in practice but still huge — e.g. ``arctanh(1 - 1e-7) ≈ 8.4`` vs.
    ``arctanh(0) = 0``), so a submission with a near-perfect Spearman on a couple of
    endpoints and ~0 on the rest could macro-average to ~0.99 instead of the
    naively-expected ~0.5, letting a handful of easy/lucky endpoints dominate the
    macro score. A plain mean — already used for ST-RAE/MAE/R2/Kendall_Tau — doesn't have
    this failure mode, so Spearman_R now uses one too, for the same reason
    Kendall_Tau always has: Fisher's z-transform has no standard extension to
    Kendall's tau (different asymptotic sampling distribution), so there was never a
    transform-based option for it here.

    Args:
        all_endpoint_bootstrap_results (pd.DataFrame): Per-endpoint bootstrap metrics
            for a single track, as returned by ``add_macro_endpoint``'s narrowing step
            (or by concatenating per-endpoint ``bootstrap_metrics(...)`` results).
            Must contain "Sample", "Endpoint", and one column per metric in
            ``metrics``.
        metrics (list[tuple[str, Callable]]): The metric list to macro-average — only
            the names are used here (e.g. ``ACTIVITY_METRICS`` or
            ``CLASSIFICATION_METRICS``).

    Returns:
        pd.DataFrame: One row per bootstrap sample, with columns "Sample",
            "Endpoint" (``MACRO_ENDPOINT_LABEL`` for every row), and the macro-averaged
            value of each metric in ``metrics`` for that sample.

    """
    grouped = all_endpoint_bootstrap_results.groupby("Sample")
    macro_results = pd.DataFrame(index=grouped.size().index)
    for metric_name, _ in metrics:
        logger.info(
            "Computing macro-averaged metric {} across bootstrap iterations",
            metric_name,
        )
        macro_results[metric_name] = grouped[metric_name].mean()
    macro_results = macro_results.reset_index()
    macro_results["Endpoint"] = MACRO_ENDPOINT_LABEL
    return macro_results


def pivot_endpoint_results_wide(by_endpoint_results: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-endpoint mean/std results into a single wide row.

    ``by_endpoint_results`` (as returned by ``average_bootstrap_results_by_endpoint``)
    has one row per endpoint (indexed by endpoint name, including the synthetic
    ``MACRO_ENDPOINT_LABEL`` ("MA") pseudo-endpoint computed by
    ``compute_macro_bootstrap_results``) and one column per ``<metric>_mean`` /
    ``<metric>_std``. This flattens it into a single-row DataFrame suitable for saving
    as a submission's ``averaged-results.parquet``, with every endpoint's columns
    consistently prefixed as ``f"{endpoint}_{metric}_{mean|std}"`` (e.g.
    ``"CYP3A4_pIC50_active_MAE_mean"``, ``"MA_ST-RAE_mean"``).

    A leaderboard for any single endpoint (macro or real) is then built by narrowing
    back down to that endpoint's columns and stripping the prefix — see
    ``aws_leaderboards._narrow_averaged_results_to_endpoint`` — so ``primary_metric``
    is always a bare metric name (e.g. ``"ST-RAE"``) regardless of which endpoint a
    given leaderboard targets.

    Args:
        by_endpoint_results (pd.DataFrame): Per-endpoint mean/std results, indexed by
            endpoint name (including ``MACRO_ENDPOINT_LABEL``).

    Returns:
        pd.DataFrame: A single-row DataFrame with one column per
            endpoint/metric/statistic combination.

    """
    wide_row: dict[str, float] = {}
    for endpoint, row in by_endpoint_results.iterrows():
        for column, value in row.items():
            wide_row[f"{endpoint}_{column}"] = value
    return pd.DataFrame([wide_row])


def average_bootstrap_results_by_endpoint(
    all_endpoint_bootstrap_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the average results of the bootstrapped samples for each endpoint.

    Args:
        all_endpoint_bootstrap_results (pd.DataFrame): A DataFrame containing the
            bootstrapped results for each endpoint.

    Returns:
        pd.DataFrame: A DataFrame containing the average results of the bootstrapped
                      samples.

    """
    logger.info("Calculating average bootstrap results by endpoint")
    agg_df = (
        all_endpoint_bootstrap_results.set_index("Sample")
        .groupby("Endpoint")
        .agg(["mean", "std"])
    )
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    return agg_df


def _metric_needs_credible_interval_bounds(metric_func: Callable) -> bool:
    """True if ``metric_func`` accepts ``y_true_upper``/``y_true_lower`` keywords.

    Lets ``bootstrap_metrics`` dispatch the credible-interval bounds only to metrics
    that use them (e.g. ``rae_soft_threshold_absolute_error``), while other metrics in
    the same list (MAE, R2, ...) keep their plain two-argument call — introspecting
    the signature avoids hardcoding metric names here.
    """
    params = inspect.signature(metric_func).parameters
    return "y_true_upper" in params and "y_true_lower" in params


def bootstrap_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    endpoint: str,
    n_bootstrap_samples: int,
    metrics: list[tuple[str, Callable]] = ACTIVITY_METRICS,
    y_true_upper: np.ndarray | None = None,
    y_true_lower: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate bootstrap metrics given predicted and true values.

    Args:
        y_pred (np.ndarray): The predicted values.
        y_true (np.ndarray): The true values.
        endpoint (str): The endpoint for which the metrics are being calculated.
        n_bootstrap_samples (int): The number of bootstrap samples to generate.
        metrics (list[tuple[str, Callable]]): The ``(name, func)`` metric list to
            compute for every bootstrap sample — ``ACTIVITY_METRICS`` for a
            regression endpoint, ``CLASSIFICATION_METRICS`` for a classification
            (TDI) endpoint. Defaults to ``ACTIVITY_METRICS``.
        y_true_upper (np.ndarray | None): Per-compound upper credible-interval bound
            for ``y_true``, aligned with ``y_true``/``y_pred``. Only consumed by
            metrics whose signature accepts ``y_true_upper``/``y_true_lower`` (see
            ``_metric_needs_credible_interval_bounds``), e.g. the soft-thresholded
            RAE metric — ignored by every other metric. Required if ``metrics``
            includes such a metric, otherwise optional.
        y_true_lower (np.ndarray | None): Per-compound lower credible-interval bound,
            counterpart to ``y_true_upper``.

    Returns:
        pd.DataFrame: A DataFrame containing the bootstrap metrics for the given
                      endpoint.

    Raises:
        RuntimeError: If a metric cannot be calculated, or returns a non-finite
            value with no entry in ``METRIC_NAN_FALLBACK``, for any bootstrap sample
            — rather than silently scoring that sample as 0 (which would misrepresent
            a real failure as a perfect score for error metrics like MAE/ST-RAE).
            Metrics listed in ``METRIC_NAN_FALLBACK`` (e.g. Spearman_R/Kendall_Tau,
            which are mathematically undefined for a zero-variance bootstrap sample —
            such as a submission predicting the same value for every compound) use
            that fallback value instead of raising. This also covers a metric that
            needs credible-interval bounds (e.g. ST-RAE) when none were supplied.

    """
    metrics_with_bounds_flag = [
        (name, func, _metric_needs_credible_interval_bounds(func))
        for name, func in metrics
    ]

    bootstrap_metrics_list = []
    for bootstrap_iteration, idx in enumerate(
        bootstrap_sampling(y_true.shape[0], n_bootstrap_samples)
    ):
        metric_values = {"Sample": bootstrap_iteration, "Endpoint": endpoint}
        for metric_name, metric_func, needs_bounds in metrics_with_bounds_flag:
            try:
                if needs_bounds:
                    if y_true_upper is None or y_true_lower is None:
                        raise ValueError(
                            f"Metric '{metric_name}' requires credible-interval "
                            "bounds (y_true_upper/y_true_lower), but none were "
                            "provided to bootstrap_metrics."
                        )
                    metric_value = metric_func(
                        y_true[idx],
                        y_pred[idx],
                        y_true_upper=y_true_upper[idx],
                        y_true_lower=y_true_lower[idx],
                    )
                else:
                    metric_value = metric_func(y_true[idx], y_pred[idx])
                if not isinstance(metric_value, (int, float)):
                    metric_value = metric_value.statistic
            except Exception as e:
                raise RuntimeError(
                    f"Error calculating metric '{metric_name}' for endpoint "
                    f"'{endpoint}' (bootstrap sample {bootstrap_iteration}): {e}"
                ) from e
            if not np.isfinite(metric_value):
                if metric_name not in METRIC_NAN_FALLBACK:
                    raise RuntimeError(
                        f"Metric '{metric_name}' for endpoint '{endpoint}' "
                        f"(bootstrap sample {bootstrap_iteration}) returned a "
                        f"non-finite value: {metric_value}"
                    )
                metric_value = METRIC_NAN_FALLBACK[metric_name]
            metric_values[metric_name] = metric_value
        bootstrap_metrics_list.append(metric_values)

    bootstrap_df = pd.DataFrame(bootstrap_metrics_list)
    return bootstrap_df


# ---------------------------------------------------------------------------
# Structure scoring
# ---------------------------------------------------------------------------


def score_single_structure(
    model_path: str,
    ref_path: str,
    max_pb_failures: int = POSEBUSTERS_MAX_FAILURES,
    smiles: str | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Score one predicted protein-ligand complex PDB against the reference.

    Runs SCRMSDScorer (BiSyRMSD + LDDT-LP) and LDDTPLIScorer (LDDT-PLI) and
    returns the top-scoring ligand pair ranked by LDDT-PLI then BiSyRMSD.

    When ``smiles`` is available, the ligands handed to both scorers are built
    from it (``posebusters_utils.bonded_ligand_molblocks``, each copy loaded
    via ``ost.io.SDFStrToEntity``) instead of a plain residue selection on the
    PDB. OST falls back to guessing bonds from interatomic distance for any
    residue not in its compound library (true for a custom ``LIG``), and two
    independently posed copies of the same molecule (the ground-truth crystal
    pose and a structure-prediction model's pose) can end up with genuinely
    different guessed bond graphs — a borderline distance, such as a
    transannular contact in a strained ring, falls on opposite sides of the
    bonding cutoff in the two conformations — which makes ligand matching
    fail outright with no assignment found even when both poses are the
    correct compound. Building both ligands from the same SMILES sidesteps
    that. When either side's complex genuinely contains more than one copy of
    the ligand (e.g. symmetry mates, multiple binding sites), every copy is
    bond-order assigned and handed to the scorers as its own entity (chain
    renamed per copy so the reconciliation keys below stay distinct) — the
    existing best-pair selection then picks whichever pairing scores best.
    Falls back to OST's own residue selection (and its multiple-copy guard)
    when no SMILES is given or bond-order assignment fails for every copy on
    either side.

    Also runs PoseBusters (dock config) on the predicted ligand pose, against
    the protein from the *predicted* complex so both are in the same
    coordinate frame. PoseBusters validates a pose, not a (ref, model)
    pairing, so — when SMILES is available — it's run once per *distinct*
    model copy that appears among the candidate pairings (memoized), and each
    pairing's scores are overridden to worst-case (LDDT-PLI=0, LDDT-LP=0,
    BiSyRMSD=``BISYRMSD_NAN_PENALTY``) if *that specific copy* fails more than
    ``max_pb_failures`` checks — this happens *before* picking the best pair,
    so an invalid copy can't win the sort just because its OST scores looked
    good, and a valid copy isn't wrongly zeroed out by some other copy's
    failure. In the naive-fallback path (no SMILES, or bond-order assignment
    failed for every copy), there's no per-copy bonded ligand to check
    individually, so PoseBusters instead validates one dedupe-picked
    representative copy (see ``posebusters_utils.dedupe_ligand_copies``)
    against the overall best pair, as this function has always done in that
    case. Bond orders are recovered from ``smiles`` when provided via
    ``posebusters_utils.assign_bond_orders_from_template`` (RDKit's PDB parser
    leaves every bond single, which breaks geometry and energy checks; the
    plain ``AllChem.AssignBondOrdersFromTemplate`` is not used directly
    because it only tries the first topological match, which can raise a
    spurious valence error or silently mis-assign a locally symmetric ring —
    see that function's docstring).
    Returns NaN for every metric if scoring fails for any reason, so that a
    single bad submission file does not abort the full evaluation.

    Args:
        model_path (str): Filesystem path to the predicted complex PDB file.
        ref_path (str): Filesystem path to the reference complex PDB file.
        max_pb_failures (int): Maximum number of PoseBusters checks the predicted
            ligand may fail before its scores are zeroed. Defaults to
            ``POSEBUSTERS_MAX_FAILURES``.
        smiles (str | None): Ground-truth SMILES for the ligand, used to build
            an unambiguous ligand for OST's scorers and to assign correct bond
            orders before running PoseBusters. When ``None``, OST guesses
            ligand connectivity from distance and PoseBusters leaves bond
            orders as parsed (all single), either of which may cause false
            failures.

    Returns:
        tuple[dict[str, float], list[str]]: A pair of (scores, failed_pb_checks).
            ``scores`` maps metric name to value (keys: ``LDDT-PLI``, ``BiSyRMSD``,
            ``LDDT-LP``); any metric that cannot be computed is ``np.nan``.
            ``failed_pb_checks`` is the sorted list of PoseBusters check names that
            failed when the failure count exceeds ``max_pb_failures``; empty when the
            pose passes, when PoseBusters is skipped due to parse errors, or when
            scoring fails entirely.

    """
    try:
        from ost import io  # type: ignore[import]
        from ost.mol.alg.ligand_scoring import (  # type: ignore[import]
            LDDTPLIScorer,
            SCRMSDScorer,
        )
        from ost.mol.alg.scoring_base import PDBPrep  # type: ignore[import]
        from rdkit import Chem

        from .posebusters_utils import bonded_ligand_copies

        # fault_tolerant=True allows loading PDBs with minor format issues
        # (missing atoms, non-standard records) that are common in docking output.
        model = PDBPrep(model_path, fault_tolerant=True)
        ref = PDBPrep(ref_path, fault_tolerant=True)

        # Prefer ligands with bond orders assigned from the ground-truth SMILES
        # (see bonded_ligand_copies) over OST's own distance-based guessing,
        # since two independently posed copies of the same molecule can end up
        # with genuinely different guessed bond graphs and make ligand matching
        # fail outright. Every copy on each side is bond-order assigned and
        # passed as its own entity (LigandScorer expands each list item's
        # residues internally, so a list of single-ligand entities works the
        # same as a list of multi-residue selections). Two independently
        # loaded SDF entities both default to the same chain/residue-number
        # identity, which would collide in the chain.residue_number
        # reconciliation keys below — renamed per copy to keep them distinct.
        # model_key_to_model_copy remembers which all-atom RDKit Mol (still
        # carrying its bond orders) backed each OST entity, keyed the same way
        # as the reconciliation below, so PoseBusters can later be run against
        # the *actual* copy a given pairing used — see the PoseBusters section.
        # Falls back to OST's naive residue selection when no SMILES is
        # available or bond-order assignment fails for every copy on either
        # side; in that fallback, model_key_to_model_copy stays empty.
        model_ligs: list = []
        ref_ligs: list = []
        model_key_to_model_copy: dict[str, Chem.Mol] = {}
        if smiles is not None:
            model_copies = bonded_ligand_copies(model_path, smiles)
            ref_copies = bonded_ligand_copies(ref_path, smiles)
            if model_copies and ref_copies:
                for i, copy in enumerate(model_copies):
                    ent = io.SDFStrToEntity(Chem.MolToMolBlock(Chem.RemoveHs(copy)))
                    ent.EditXCS().RenameChain(ent.chains[0], f"LIG{i}")
                    model_ligs.append(ent)
                    residue = ent.residues[0]
                    model_key_to_model_copy[
                        f"{residue.chain.name}.{residue.number}"
                    ] = copy
                for i, copy in enumerate(ref_copies):
                    ent = io.SDFStrToEntity(Chem.MolToMolBlock(Chem.RemoveHs(copy)))
                    ent.EditXCS().RenameChain(ent.chains[0], f"LIG{i}")
                    ref_ligs.append(ent)

        if not model_ligs or not ref_ligs:
            # Select only heavy atoms of the LIG residue. Hydrogens are excluded
            # because protonation state varies between docking programs and
            # crystal structures and would cause spurious subgraph isomorphism
            # failures.
            model_ligs = [model.Select("rname=LIG and ele!=H")]
            ref_ligs = [ref.Select("rname=LIG and ele!=H")]

            n_model_ligs = len(model_ligs[0].residues)
            if n_model_ligs > 2:
                logger.warning(
                    "Model {} contains {} LIG residues (expected at most 2)"
                    " — returning NaN scores.",
                    model_path,
                    n_model_ligs,
                )
                return dict(_NAN_STRUCTURE_METRICS), []

        logger.info("Scoring structure {} against reference {}", model_path, ref_path)

        # substructure_match=True allows the model ligand to be a subgraph of
        # the reference (handles partial docking output). coverage_delta=0.1
        # permits up to 10% atom coverage difference before the pair is
        # considered unmatchable.
        scrmsd_sc = SCRMSDScorer(
            model=model,
            target=ref,
            model_ligands=model_ligs,
            target_ligands=ref_ligs,
            substructure_match=True,
            coverage_delta=0.1,
        )
        lddt_pli_sc = LDDTPLIScorer(
            model=model,
            target=ref,
            model_ligands=model_ligs,
            target_ligands=ref_ligs,
            substructure_match=True,
            coverage_delta=0.1,
        )

        # Each scorer solves the protein chain mapping and ligand assignment
        # problem independently. Integer indices are NOT comparable across
        # scorers for structures with multiple equivalent ligand copies.
        # Use "chain.residue_number" string keys as stable ligand identifiers
        # and join post-hoc.
        def _lig_key(lig: object) -> str:
            return f"{lig.chain.name}.{lig.number}"  # type: ignore[attr-defined]

        pli_ref_keys = [_lig_key(l) for l in lddt_pli_sc.target_ligands]
        pli_mdl_keys = [_lig_key(l) for l in lddt_pli_sc.model_ligands]
        sc_ref_keys = [_lig_key(l) for l in scrmsd_sc.target_ligands]
        sc_mdl_keys = [_lig_key(l) for l in scrmsd_sc.model_ligands]

        # LDDT-PLI assignment is the primary source of truth for which
        # (ref ligand, model ligand) pairs to report.
        results = []
        for pli_i, pli_j in lddt_pli_sc.assignment:
            ref_key = pli_ref_keys[pli_i]
            mdl_key = pli_mdl_keys[pli_j]

            if ref_key not in sc_ref_keys or mdl_key not in sc_mdl_keys:
                logger.warning(
                    "LDDT-PLI pair ({}, {}) not found in SCRMSDScorer ligand lists"
                    " for {} — skipping pair.",
                    ref_key,
                    mdl_key,
                    model_path,
                )
                continue
            sc_i = sc_ref_keys.index(ref_key)
            sc_j = sc_mdl_keys.index(mdl_key)

            # state_matrix encodes why a given (ref, model) pair could not be
            # scored. State 0 means the score is valid.
            if scrmsd_sc.state_matrix[sc_i, sc_j] != 0:
                state = scrmsd_sc.state_matrix[sc_i, sc_j]
                decoding = scrmsd_sc.state_decoding[state]
                logger.warning(
                    "SCRMSDScorer state for ({}, {}) is {} ({}) —"
                    " BiSyRMSD/LDDT-LP will be NaN.",
                    ref_key,
                    mdl_key,
                    state,
                    decoding,
                )
                results.append(
                    {
                        "LDDT-PLI": np.float64(lddt_pli_sc.score_matrix[pli_i, pli_j]),
                        "BiSyRMSD": np.nan,
                        "LDDT-LP": np.nan,
                        "_model_copy": model_key_to_model_copy.get(mdl_key),
                    }
                )
            else:
                aux = scrmsd_sc.aux_matrix[sc_i, sc_j]
                results.append(
                    {
                        "LDDT-PLI": np.float64(lddt_pli_sc.score_matrix[pli_i, pli_j]),
                        "BiSyRMSD": np.float64(scrmsd_sc.score_matrix[sc_i, sc_j]),
                        "LDDT-LP": np.float64(aux["lddt_lp"]),
                        "_model_copy": model_key_to_model_copy.get(mdl_key),
                    }
                )

        if not results:
            logger.warning(
                "No ligand assignment found between {} and {} — returning NaN scores."
                " LDDT-PLI assignment: {}. SCRMSDScorer assignment: {}.",
                model_path,
                ref_path,
                lddt_pli_sc.assignment,
                scrmsd_sc.assignment,
            )
            return dict(_NAN_STRUCTURE_METRICS), []

        from posebusters import PoseBusters  # type: ignore[import]

        from .posebusters_utils import (
            assign_bond_orders_from_template,
            dedupe_ligand_copies,
        )

        # PoseBusters checks physical/geometric validity of *one* predicted
        # ligand pose against the *predicted* receptor (it never touches the
        # reference), so it's a property of a model copy, not of a
        # (ref, model) pairing — run it once per distinct model copy that
        # appears in `results` (memoized by object identity) and apply the
        # worst-case override to every pairing using that copy, *before*
        # picking the best pair. This is what makes an invalid copy lose the
        # sort on its own merits, rather than best_result being chosen first
        # and then checked against a possibly-different, arbitrarily-picked
        # copy (which could wrongly validate an invalid winner, or wrongly
        # zero out a valid one because some other copy failed).
        pred_mol = Chem.MolFromPDBFile(model_path, removeHs=False, sanitize=False)
        pb_protein = None
        if pred_mol is not None:
            fragments = Chem.SplitMolByPDBResidues(pred_mol)
            fragments.pop("LIG", None)
            for frag in fragments.values():
                pb_protein = (
                    frag if pb_protein is None else Chem.CombineMols(pb_protein, frag)
                )

        def _posebusters_failed_checks(ligand: Chem.Mol) -> list[str]:
            template = Chem.MolFromSmiles(smiles) if smiles is not None else None
            pb_sc = PoseBusters(config="dock")
            pb_result = pb_sc.bust(
                mol_pred=ligand, mol_true=template, mol_cond=pb_protein
            )
            pb_bool_cols = [c for c in pb_result.columns if pb_result[c].dtype == bool]
            row = pb_result[pb_bool_cols].iloc[0]
            return sorted(row.index[~row].tolist())

        if pred_mol is None or pb_protein is None:
            logger.warning(
                "RDKit could not parse {} (or find its protein) for PoseBusters —"
                " skipping check, returning OST scores.",
                model_path,
            )
            for r in results:
                r.pop("_model_copy", None)
            results.sort(key=lambda r: (-r["LDDT-PLI"], r["BiSyRMSD"]))
            return results[0], []

        if any(r["_model_copy"] is not None for r in results):
            # SMILES-available path: every pairing already knows exactly which
            # bonded model copy it used (see model_key_to_model_copy above).
            pb_cache: dict[int, list[str]] = {}
            for r in results:
                copy = r.pop("_model_copy")
                if copy is None:
                    continue
                failed_checks = pb_cache.setdefault(
                    id(copy), _posebusters_failed_checks(copy)
                )
                if len(failed_checks) > max_pb_failures:
                    logger.warning(
                        "Model {} copy failed {} PoseBusters checks (max allowed {})"
                        " — zeroing that pairing's scores.",
                        model_path,
                        len(failed_checks),
                        max_pb_failures,
                    )
                    r["LDDT-PLI"] = 0.0
                    r["BiSyRMSD"] = BISYRMSD_NAN_PENALTY
                    r["LDDT-LP"] = 0.0
                r["_pb_failures"] = failed_checks
            results.sort(key=lambda r: (-r["LDDT-PLI"], r["BiSyRMSD"]))
            best_result = results[0]
            failed_checks = best_result.pop("_pb_failures")
            return best_result, failed_checks if len(
                failed_checks
            ) > max_pb_failures else []

        # Naive-fallback path (no SMILES, or bond-order assignment failed for
        # every copy): no per-copy bonded Mols exist to check individually, so
        # fall back to a single dedupe-based check against the overall best
        # pair — the same behavior this function has always had for this case.
        for r in results:
            r.pop("_model_copy", None)
        results.sort(key=lambda r: (-r["LDDT-PLI"], r["BiSyRMSD"]))
        best_result = results[0]

        pb_ligand = Chem.SplitMolByPDBResidues(pred_mol).get("LIG")
        if pb_ligand is None:
            logger.warning(
                "No LIG residue in {} for PoseBusters — skipping check, returning OST scores.",
                model_path,
            )
            return best_result, []
        pb_ligand = dedupe_ligand_copies(pb_ligand)
        if smiles is not None:
            template = Chem.MolFromSmiles(smiles)
            if template is not None:
                try:
                    pb_ligand = assign_bond_orders_from_template(template, pb_ligand)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Bond order assignment failed for {} ({}) — PoseBusters will use single-bond ligand.",
                        model_path,
                        exc,
                    )

        failed_checks = _posebusters_failed_checks(pb_ligand)
        if len(failed_checks) > max_pb_failures:
            logger.warning(
                "Model {} failed {} PoseBusters checks (max allowed {}) — zeroing scores.",
                model_path,
                len(failed_checks),
                max_pb_failures,
            )
            return {
                "LDDT-PLI": 0.0,
                "BiSyRMSD": BISYRMSD_NAN_PENALTY,
                "LDDT-LP": 0.0,
            }, failed_checks
        return best_result, []

    except Exception as e:  # noqa: BLE001
        logger.exception("OST scoring failed for {} vs {}: {}", model_path, ref_path, e)
        return dict(_NAN_STRUCTURE_METRICS), []


def dummy_score_single_structure(
    model_path: str, ref_path: str
) -> tuple[dict[str, float], list[str]]:
    """Dummy scoring function that returns random scores for testing."""
    import random

    return {
        "LDDT-PLI": random.uniform(0, 1),
        "BiSyRMSD": random.uniform(0, 5),
        "LDDT-LP": random.uniform(0, 1),
    }, []


def score_structure_predictions(
    predicted_structures: dict[str, str],
    ground_truth_structures: dict[str, str],
    max_pb_failures: int = POSEBUSTERS_MAX_FAILURES,
    smiles_map: pd.DataFrame | dict[str, str] | None = None,
    id_column: str = "Molecule_Name",
    smiles_column: str = "SMILES",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Score all predicted protein-ligand complex PDB files against ground truth.

    Iterates over ``predicted_structures``, skipping any molecule ID not found
    in ``ground_truth_structures``. A ``coverage`` column (1.0 = matched,
    0.0 = failed) is added before NaN values are replaced with worst-case
    penalties so every compound contributes to bootstrap aggregation:

    - LDDT-PLI, LDDT-LP (↑ better): NaN → 0.0
    - BiSyRMSD (↓ better): NaN → ``BISYRMSD_NAN_PENALTY``

    Poses that fail more than ``max_pb_failures`` PoseBusters checks are zeroed
    before aggregation (LDDT-PLI=0, LDDT-LP=0, BiSyRMSD=``BISYRMSD_NAN_PENALTY``);
    those compounds still count as covered (coverage=1.0) since OST scored them.

    Args:
        predicted_structures (dict[str, str]): Mapping from molecule ID to
            filesystem path of the predicted PDB file.
        ground_truth_structures (dict[str, str]): Mapping from molecule ID
            to filesystem path of the reference PDB file.
        max_pb_failures (int): Maximum number of PoseBusters checks a ligand
            may fail before its scores are zeroed. Forwarded to
            ``score_single_structure``. Defaults to ``POSEBUSTERS_MAX_FAILURES``.
        smiles_map (pd.DataFrame | dict[str, str] | None): Ground-truth SMILES
            for each molecule, forwarded to ``score_single_structure`` to build
            an unambiguous ligand for OST's scorers and for bond-order
            assignment before PoseBusters runs (see that function's
            docstring). Either a DataFrame with an ``id_column`` and a
            ``smiles_column`` (duplicate IDs keep their first SMILES), or an
            already-built ``{molecule_id: smiles}`` dict. When ``None``, OST
            guesses ligand connectivity from distance and PoseBusters leaves
            bond orders as parsed from PDB (all single).
        id_column (str): Column in ``smiles_map`` holding molecule IDs, used
            only when ``smiles_map`` is a DataFrame. Defaults to
            ``"Molecule_Name"``.
        smiles_column (str): Column in ``smiles_map`` holding SMILES, used
            only when ``smiles_map`` is a DataFrame. Defaults to ``"SMILES"``.

    Returns:
        tuple[pd.DataFrame, dict[str, list[str]]]: A pair of
            (per_compound_df, pb_failures) where per_compound_df has columns
            ``Molecule_Name``, ``LDDT-PLI``, ``BiSyRMSD``, ``LDDT-LP``,
            ``coverage`` (1.0 if successfully matched, 0.0 otherwise), and
            pb_failures maps molecule ID to the sorted list of PoseBusters
            check names that caused its scores to be zeroed. Only compounds that
            exceed ``max_pb_failures`` appear in pb_failures.

    """
    if isinstance(smiles_map, pd.DataFrame):
        smiles_map = (
            smiles_map[[id_column, smiles_column]]
            .dropna(subset=[smiles_column])
            .drop_duplicates(subset=[id_column])
            .set_index(id_column)[smiles_column]
            .to_dict()
        )

    logger.info(
        "Scoring {} structure predictions against ground truth",
        len(predicted_structures),
    )
    rows = []
    pb_failures: dict[str, list[str]] = {}
    for mol_id, model_path in predicted_structures.items():
        if mol_id not in ground_truth_structures:
            logger.warning("No ground truth found for {}, skipping.", mol_id)
            continue
        scores, failed_checks = score_single_structure(
            model_path,
            ground_truth_structures[mol_id],
            max_pb_failures=max_pb_failures,
            smiles=smiles_map.get(mol_id) if smiles_map is not None else None,
        )
        rows.append({"Molecule_Name": mol_id, **scores})
        if failed_checks:
            pb_failures[mol_id] = failed_checks

    per_compound_df = pd.DataFrame(rows)
    n_scored = int(per_compound_df["LDDT-PLI"].notna().sum())
    logger.info(
        "Successfully scored {}/{} structures.",
        n_scored,
        len(predicted_structures),
    )

    # Record coverage before filling so the per-compound parquet is transparent
    per_compound_df["coverage"] = per_compound_df["LDDT-PLI"].notna().astype(float)

    # Apply worst-case penalties so every compound contributes to the bootstrap mean
    per_compound_df["LDDT-PLI"] = per_compound_df["LDDT-PLI"].fillna(0.0)
    per_compound_df["LDDT-LP"] = per_compound_df["LDDT-LP"].fillna(0.0)
    per_compound_df["BiSyRMSD"] = per_compound_df["BiSyRMSD"].fillna(
        BISYRMSD_NAN_PENALTY
    )

    return per_compound_df, pb_failures


def bootstrap_structure_metrics(
    per_compound_df: pd.DataFrame,
    n_bootstrap_samples: int,
) -> pd.DataFrame:
    """Bootstrap aggregate structure metrics over the set of scored compounds.

    Each bootstrap iteration resamples the compounds (rows) with replacement
    and computes the mean of each metric. Failed structures are already filled
    with worst-case penalty values by ``score_structure_predictions``, so
    ``np.mean`` is used — every compound contributes to the aggregate.
    The returned DataFrame uses the same ``Sample`` / ``Endpoint`` schema as
    ``bootstrap_metrics``, so ``average_bootstrap_results_by_endpoint`` can be
    reused unchanged. Coverage is a scalar property of the whole submission and
    is not bootstrapped — it is added to ``averaged_df`` separately in
    ``score_structure_submission``.

    Args:
        per_compound_df (pd.DataFrame): Output of ``score_structure_predictions``
            — one row per compound, columns ``LDDT-PLI``, ``BiSyRMSD``,
            ``LDDT-LP``, ``coverage``.
        n_bootstrap_samples (int): Number of bootstrap iterations.

    Returns:
        pd.DataFrame: Bootstrap results with columns:
            ``Sample``, ``Endpoint``, ``LDDT-PLI``, ``BiSyRMSD``, ``LDDT-LP``.

    """
    _BOOTSTRAP_COLS = STRUCTURE_METRICS
    scores = per_compound_df[_BOOTSTRAP_COLS].to_numpy()
    n_compounds = scores.shape[0]

    rows = []
    for sample_idx, idx in enumerate(
        bootstrap_sampling(n_compounds, n_bootstrap_samples)
    ):
        sample_means = np.mean(scores[idx], axis=0)
        row: dict[str, object] = {"Sample": sample_idx, "Endpoint": "Structure"}
        for col, value in zip(_BOOTSTRAP_COLS, sample_means):
            row[col] = value
        rows.append(row)

    return pd.DataFrame(rows)
