from __future__ import annotations

import zipfile
import tempfile
import MDAnalysis as mda
from pathlib import Path
from typing import Union
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


STRUCTURE_DATASET_SIZE = 20


def _residue_icode(residue) -> str:
    """Return a residue's insertion code, or "" if the topology has none.

    MDAnalysis raises a non-``AttributeError`` exception (``NoDataError``) when
    a topology attribute wasn't parsed at all, so ``getattr(..., default)``
    alone won't catch it.
    """
    try:
        return str(residue.icode).strip()
    except Exception:
        return ""


def _check_atom_residue_numbering(universe: "mda.Universe", name: str) -> list[str]:
    """Flag residue/atom numbering issues known to break OpenStructure scoring.

    Cofolding tools (Boltz-2 and others) occasionally emit PDBs with numbering
    that OpenStructure's residue/atom mapping can't disambiguate:

    - Duplicate (chain, resid, icode) triples labelling more than one residue
      in the same chain — breaks OST's sequence-to-structure alignment.
    - Duplicate atom names within a single residue — breaks OST's
      symmetry-corrected atom matching for RMSD/LDDT scoring.
    - Duplicate atom serial numbers across the whole structure — technically
      invalid PDB, and a common artifact of per-chain serial resets.

    Args:
        universe (mda.Universe): Parsed structure for one submitted PDB.
        name (str): Filename, used to prefix error messages.

    Returns:
        list[str]: Validation error messages (empty if numbering looks clean).

    """
    numbering_errors: list[str] = []

    for segment in universe.segments:
        seen_residue_keys: dict[tuple[int, str], str] = {}
        for residue in segment.residues:
            key = (int(residue.resid), _residue_icode(residue))
            if key in seen_residue_keys:
                resid, icode = key
                label = f"{resid}{icode}" if icode else str(resid)
                numbering_errors.append(
                    f"{name}: Duplicate residue number {label} in chain "
                    f"'{segment.segid}' (both {seen_residue_keys[key]!r} and "
                    f"{residue.resname!r} use it)"
                )
            else:
                seen_residue_keys[key] = residue.resname

            atom_names = list(residue.atoms.names)
            if len(atom_names) != len(set(atom_names)):
                duplicate_names = sorted({n for n in atom_names if atom_names.count(n) > 1})
                numbering_errors.append(
                    f"{name}: Residue {residue.resname}{residue.resid} in chain "
                    f"'{segment.segid}' has duplicate atom name(s): {duplicate_names}"
                )

    if hasattr(universe.atoms, "ids"):
        atom_ids = list(universe.atoms.ids)
        if len(atom_ids) != len(set(atom_ids)):
            n_duplicates = len(atom_ids) - len(set(atom_ids))
            numbering_errors.append(
                f"{name}: Found {n_duplicates} duplicated atom serial number(s) "
                "across the structure (expected unique atom serials file-wide)"
            )

    return numbering_errors


def validate_structure_submission(
    structure_predictions_file: Union[str, Path],
    expected_ids: set[str] | None = None,
    expected_ligand_smiles: dict[str, str] | None = None,
    require_lig_resname: bool = True,
    check_numbering: bool = True,
) -> tuple[bool, list[str]]:
    """Validate a CYP structure-track submission (a zip of predicted complex PDBs).

    Args:
        structure_predictions_file (str | Path): Path to the submitted zip archive.
        expected_ids (set[str] | None): Expected structure IDs (PDB filename stems).
            If provided, the submission is checked for missing IDs instead of a
            fixed file count.
        expected_ligand_smiles (dict[str, str] | None): Mapping from structure ID to
            the expected ligand SMILES. When provided, each submitted ligand's
            connectivity is checked against this template via RDKit
            ``AssignBondOrdersFromTemplate``.
        require_lig_resname (bool): Whether to require the ligand residue in each
            PDB to be named exactly ``LIG`` (and be the only such residue).
        check_numbering (bool): Whether to check for residue/atom numbering
            issues (duplicate residue numbers, duplicate atom names within a
            residue, duplicate atom serials) that are known to make
            OpenStructure's scoring fail on some cofolding tool outputs.

    Returns:
        tuple[bool, list[str]]: Whether the submission is valid, and a list of
            validation error messages (empty if valid).

    """
    errors: list[str] = []
    path = Path(structure_predictions_file)

    if not path.exists():
        return False, [f"File does not exist: {path}"]

    if path.suffix.lower() != ".zip":
        return False, ["Structure predictions file must be a .zip file."]

    try:
        with zipfile.ZipFile(path, "r") as zip_file:
            pdb_files = [name for name in zip_file.namelist() if name.lower().endswith(".pdb")]

            if not pdb_files:
                return False, ["Zip file contains no PDB files."]

            # --- ID Consistency Checks ---
            submitted_ids = {Path(name).stem for name in pdb_files}
            if expected_ids is not None:
                expected_ids = {str(x) for x in expected_ids}
                missing = sorted(expected_ids - submitted_ids)
                if missing:
                    errors.append(f"Missing {len(missing)} expected structure(s): {missing[:20]}")
            elif len(pdb_files) != STRUCTURE_DATASET_SIZE:
                errors.append(
                    f"Zip file contains {len(pdb_files)} .pdb files, expected {STRUCTURE_DATASET_SIZE}."
                )

            # --- MDAnalysis Structural Checks ---
            if require_lig_resname or check_numbering:
                with tempfile.TemporaryDirectory() as tmpdir:
                    for name in pdb_files:
                        # Extract to temp file so MDAnalysis can read it
                        tmp_path = zip_file.extract(name, path=tmpdir)

                        try:
                            # Suppress warnings for missing chain IDs/elements if necessary
                            u = mda.Universe(tmp_path)

                            # 0. Residue/atom numbering issues that break OpenStructure scoring
                            # (checked independently of require_lig_resname, and before the LIG
                            # checks below so a missing-LIG `continue` doesn't skip it).
                            if check_numbering:
                                errors.extend(_check_atom_residue_numbering(u, name))

                            if not require_lig_resname:
                                continue

                            # 1. Check for residue name 'LIG'
                            ligands = u.select_atoms("resname LIG")
                            if len(ligands) == 0:
                                errors.append(f"{name}: Missing residue 'LIG'")
                                continue

                            # 2. Ensure only ONE residue named LIG exists
                            if len(ligands.residues) > 1:
                                errors.append(f"{name}: Found {len(ligands.residues)} 'LIG' residues, expected 1")

                            # 3. Ensure at most three chains exist in the entire PDB: protein,
                            # heme cofactor, and ligand. Unlike PXR's protein+ligand-only
                            # submissions, every CYP isoform is a heme-containing enzyme, so a
                            # correctly folded complex has one extra chain for the HEM cofactor.
                            if len(u.segments) > 3:
                                errors.append(
                                    f"{name}: Found {len(u.segments)} chains, expected 3 or fewer "
                                    "(protein + heme cofactor + ligand)"
                                )

                            # 4. Check ligand graph matches expected SMILES connectivity
                            if expected_ligand_smiles is not None:
                                pdb_id = Path(name).stem
                                expected_smi = expected_ligand_smiles.get(pdb_id)
                                if expected_smi is not None:
                                    ref_mol = Chem.MolFromSmiles(expected_smi)
                                    if ref_mol is None:
                                        errors.append(
                                            f"{name}: Could not parse expected SMILES for '{pdb_id}'"
                                        )
                                    else:
                                        lig_pdb_path = Path(tmpdir) / f"{pdb_id}_lig.pdb"
                                        ligands.write(str(lig_pdb_path))
                                        lig_mol = Chem.MolFromPDBFile(
                                            str(lig_pdb_path), removeHs=True, sanitize=False
                                        )
                                        if lig_mol is None:
                                            errors.append(
                                                f"{name}: RDKit could not parse LIG residue"
                                            )
                                        else:
                                            try:
                                                RDLogger.DisableLog("rdApp.*")
                                                AllChem.AssignBondOrdersFromTemplate(ref_mol, lig_mol)
                                                RDLogger.EnableLog("rdApp.*")
                                            except ValueError:
                                                RDLogger.EnableLog("rdApp.*")
                                                errors.append(
                                                    f"{name}: Ligand connectivity does not match "
                                                    f"expected SMILES '{expected_smi}'"
                                                )

                        except Exception as e:
                            errors.append(f"{name}: MDAnalysis failed to parse file: {e}")

    except zipfile.BadZipFile:
        return False, ["File is not a valid zip archive."]
    except Exception as exc:
        return False, [f"Unexpected error during validation: {exc}"]

    return len(errors) == 0, errors
