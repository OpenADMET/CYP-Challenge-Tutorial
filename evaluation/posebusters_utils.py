"""Helpers for preparing a predicted ligand pose for PoseBusters checks and
for OST's ligand scorers.

Ported from two OpenADMET sibling repos' PoseBusters scripts:
``dedupe_ligand_copies`` from ``pxr-challenge-posebusters-check/posebusters_check.py``,
``assign_bond_orders_from_template`` from
``cyp-challenge-splits/structure/posebusters_check.py``.
"""

from loguru import logger
from rdkit import Chem


def dedupe_ligand_copies(ligand: Chem.Mol, ligand_resname: str = "LIG") -> Chem.Mol:
    """Keep only the largest copy of a ligand that appears more than once.

    Ground truth crystal structures often contain several copies of the same
    ligand in the pocket (symmetry mates or multiple binding sites, all sharing
    the same residue name). ``Chem.SplitMolByPDBResidues`` groups by residue
    name, so those copies come back as one disconnected molecule; only the
    largest connected copy is kept so downstream template matching has a 1:1
    atom count.

    Args:
        ligand (Chem.Mol): Ligand fragment extracted via
            ``Chem.SplitMolByPDBResidues`` (may contain more than one
            disconnected copy).
        ligand_resname (str): Residue name the ligand was extracted under,
            used only for the warning message when more than one copy is
            found.

    Returns:
        Chem.Mol: ``ligand`` unchanged if it was a single connected molecule,
            otherwise the largest disconnected copy by heavy-atom count.

    """
    copies = Chem.GetMolFrags(ligand, asMols=True, sanitizeFrags=False)
    if len(copies) > 1:
        largest = max(copies, key=lambda m: m.GetNumHeavyAtoms())
        logger.warning(
            "{} '{}' copies in pocket, keeping the largest ({} heavy atoms)",
            len(copies),
            ligand_resname,
            largest.GetNumHeavyAtoms(),
        )
        return largest
    return ligand


def assign_bond_orders_from_template(refmol: Chem.Mol, mol: Chem.Mol) -> Chem.Mol:
    """Like ``AllChem.AssignBondOrdersFromTemplate``, but tries every
    topological match instead of just the first one, and matches on heavy
    atoms only.

    ``AssignBondOrdersFromTemplate`` strips bond orders from both molecules
    to find a topological (connectivity-only) match, then arbitrarily keeps
    whichever match ``GetSubstructMatches`` happens to return first. Two
    problems show up on ``mol`` built from a PDB with explicit hydrogens:

    * A locally symmetric ring or chain (e.g. an NH-pyrrole, which looks the
      same read clockwise or counterclockwise once bond orders are stripped)
      can have more than one topologically valid match, and only some of
      them reproduce a chemically valid Kekule structure once the template's
      bond orders are mapped back on.
    * ``refmol`` (from SMILES) represents each hydrogen as an implicit count
      on its heavy atom, while ``mol`` (from the PDB) carries them as real,
      separate atoms. Copying ``refmol``'s ``NumExplicitHs`` (usually 0)
      onto a ``mol`` atom that already has a real explicit-hydrogen neighbor
      double-counts that hydrogen during sanitization, which can force an
      extra bond order onto the atom (e.g. an aromatic nitrogen ending up
      with two double bonds instead of zero) and raise a valence error such
      as "Explicit valence ... greater than permitted" even though the
      extracted ligand is the correct compound.
    * With no ``CONECT`` records, RDKit guesses ``mol``'s bonds from
      interatomic distances alone, which can add a spurious extra bond
      across a strained small ring (e.g. a transannular 1,3 contact in a
      four-membered ring sitting closer than a real bond usually would).
      ``GetSubstructMatch`` only checks that every bond in ``refmol`` is
      present in ``mol`` — it doesn't reject ``mol`` having *more* bonds
      than that — so the match still succeeds, and the spurious bond is
      left as an untouched single bond, again raising a valence error on
      the atom that gained the unwanted extra connection.

    This works around all three: matching and bond-order assignment happen
    on a hydrogen-stripped copy of ``mol`` (trying each candidate
    topological match in turn, discarding any bond between matched atoms
    that isn't part of ``refmol``'s own connectivity before assigning bond
    orders, and keeping the first candidate that sanitizes), and the result
    is mapped back onto the original hydrogen-inclusive molecule by
    heavy-atom identity rather than by index, since removing hydrogens can
    renumber atoms.

    Args:
        refmol (Chem.Mol): The template molecule (bond orders correct, from
            SMILES).
        mol (Chem.Mol): The molecule to assign bond orders to (bond orders
            unreliable, from a PDB with no ``CONECT`` records; may include
            explicit Hs).

    Returns:
        Chem.Mol: A copy of ``mol`` (hydrogens included, with their original
            3D positions) with bond orders and aromaticity assigned from
            ``refmol``.

    Raises:
        ValueError: If no topological match exists, or none of the matches
            found sanitize successfully.

    """
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp("_origIdx", atom.GetIdx())
    mol_noh = Chem.RemoveHs(mol, sanitize=False)
    heavy_to_orig = [atom.GetIntProp("_origIdx") for atom in mol_noh.GetAtoms()]

    refmol2 = Chem.Mol(refmol)
    mol2 = Chem.Mol(mol_noh)

    # Already matching with bond orders intact?
    if not mol2.GetSubstructMatch(refmol2):
        # Flatten both molecules to find a topological (bond-order-agnostic) match.
        for template_mol in (refmol2, mol2):
            for bond in template_mol.GetBonds():
                bond.SetBondType(Chem.BondType.SINGLE)
                bond.SetIsAromatic(False)
            for atom in template_mol.GetAtoms():
                atom.SetFormalCharge(0)

        matches = mol2.GetSubstructMatches(refmol2, uniquify=False)
        if not matches:
            raise ValueError("No matching found")

        assigned_heavy = None
        errors = []
        for matching in matches:
            candidate = Chem.RWMol(mol_noh)
            matched_atoms = set(matching)
            ref_edges = {
                frozenset(
                    (matching[bond.GetBeginAtomIdx()], matching[bond.GetEndAtomIdx()])
                )
                for bond in refmol.GetBonds()
            }
            # Drop any bond between two matched atoms that has no counterpart
            # in refmol's own connectivity: a spurious extra bond guessed
            # from atomic distances, not a real one.
            for bond in list(candidate.GetBonds()):
                atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (
                    atom1 in matched_atoms
                    and atom2 in matched_atoms
                    and frozenset((atom1, atom2)) not in ref_edges
                ):
                    candidate.RemoveBond(atom1, atom2)
            for bond in refmol.GetBonds():
                atom1 = matching[bond.GetBeginAtomIdx()]
                atom2 = matching[bond.GetEndAtomIdx()]
                if candidate.GetBondBetweenAtoms(atom1, atom2) is None:
                    candidate.AddBond(atom1, atom2, bond.GetBondType())
                candidate_bond = candidate.GetBondBetweenAtoms(atom1, atom2)
                candidate_bond.SetBondType(bond.GetBondType())
                candidate_bond.SetIsAromatic(bond.GetIsAromatic())
            for atom in refmol.GetAtoms():
                candidate_atom = candidate.GetAtomWithIdx(matching[atom.GetIdx()])
                candidate_atom.SetHybridization(atom.GetHybridization())
                candidate_atom.SetIsAromatic(atom.GetIsAromatic())
                candidate_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                candidate_atom.SetFormalCharge(atom.GetFormalCharge())
            candidate = candidate.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except Exception as exc:  # noqa: BLE001 - try the next candidate match
                errors.append(str(exc))
                continue
            assigned_heavy = candidate
            break

        if assigned_heavy is None:
            raise ValueError(
                f"No topological match could be sanitized ({len(matches)} "
                f"candidate(s) tried): {'; '.join(errors)}"
            )
        mol_noh = assigned_heavy

    # If mol had no explicit hydrogens to begin with, mol_noh already *is*
    # mol (RemoveHs was a no-op) and is already sanitized — skip rebuilding
    # and re-sanitizing a fresh copy. That second sanitize pass is otherwise
    # redundant (the Chem.RWMol reconstruction can perturb RDKit's internal
    # ring-perception state, which is enough to flip Kekulization from
    # succeeding to failing on a locally symmetric ring even though nothing
    # chemically changed). Structure-prediction tools like Boltz-2 and
    # OpenFold3 output heavy-atom-only coordinates, so this is the common
    # case for their predictions, as opposed to a crystal structure PDB with
    # real deposited hydrogens.
    if mol.GetNumAtoms() == mol_noh.GetNumAtoms():
        # Sanitizing is a no-op when the per-candidate loop above already did
        # it; it's still needed here for the "already matching" fast path,
        # which never sanitizes mol_noh.
        Chem.SanitizeMol(mol_noh)
        return mol_noh

    # Transfer the heavy-atom bond orders/aromaticity back onto the original,
    # hydrogen-inclusive molecule (mapped by heavy-atom identity, since
    # RemoveHs can renumber atoms) and sanitize that.
    heavy_orig_atoms = set(heavy_to_orig)
    heavy_bonds = {
        frozenset(
            (heavy_to_orig[bond.GetBeginAtomIdx()], heavy_to_orig[bond.GetEndAtomIdx()])
        )
        for bond in mol_noh.GetBonds()
    }
    result = Chem.RWMol(mol)
    # Drop any heavy-heavy bond that didn't survive onto mol_noh (e.g. a
    # spurious bond pruned above): it needs removing here too, since this is
    # a fresh copy of the original, unpruned `mol`.
    for bond in list(result.GetBonds()):
        atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (
            atom1 in heavy_orig_atoms
            and atom2 in heavy_orig_atoms
            and frozenset((atom1, atom2)) not in heavy_bonds
        ):
            result.RemoveBond(atom1, atom2)
    for bond in mol_noh.GetBonds():
        atom1 = heavy_to_orig[bond.GetBeginAtomIdx()]
        atom2 = heavy_to_orig[bond.GetEndAtomIdx()]
        if result.GetBondBetweenAtoms(atom1, atom2) is None:
            result.AddBond(atom1, atom2, bond.GetBondType())
        result_bond = result.GetBondBetweenAtoms(atom1, atom2)
        result_bond.SetBondType(bond.GetBondType())
        result_bond.SetIsAromatic(bond.GetIsAromatic())
    for atom in mol_noh.GetAtoms():
        result_atom = result.GetAtomWithIdx(heavy_to_orig[atom.GetIdx()])
        result_atom.SetHybridization(atom.GetHybridization())
        result_atom.SetIsAromatic(atom.GetIsAromatic())
        result_atom.SetFormalCharge(atom.GetFormalCharge())
    result = result.GetMol()
    Chem.SanitizeMol(result)
    return result


def bonded_ligand_copies(
    pdb_path: str, smiles: str, ligand_resname: str = "LIG"
) -> list[Chem.Mol]:
    """Extract every copy of a complex's ligand with correct bond orders assigned.

    OST's ligand scorers (``SCRMSDScorer``/``LDDTPLIScorer``) fall back to
    guessing bonds from interatomic distance for any residue not in their
    compound library — true for a custom ``LIG`` residue. Two independently
    posed copies of the *same* molecule (e.g. a ground truth crystal pose and
    a structure-prediction model's pose) can then end up with genuinely
    different guessed bond graphs, which makes the scorer's
    ligand-matching step fail outright with no assignment found even though
    both poses are the same compound.

    This sidesteps that: every disconnected copy of the ligand residue is
    extracted (unlike :func:`dedupe_ligand_copies`, which keeps only the
    largest — a shortcut only valid when the caller genuinely wants one
    representative pose, not every copy) and independently has its bond
    orders assigned from ``smiles`` (see
    :func:`assign_bond_orders_from_template`). A crystal structure or
    predicted assembly can genuinely contain more than one copy of the same
    ligand (symmetry mates, multiple binding sites) — callers should hand
    every copy to the scorer/validity check so best-pair selection can pick
    the correct correspondence, rather than guessing which copy matters
    upfront.

    Each returned ``Mol`` keeps its explicit hydrogens and original PDB
    residue info (chain id, residue number — verified to survive
    :func:`assign_bond_orders_from_template`'s reconstruction), so a caller
    that needs an OST-ready, heavy-atom-only ligand can do
    ``Chem.RemoveHs(copy)`` + ``Chem.MolToMolBlock(...)`` itself, while a
    caller needing an all-atom ligand (e.g. for PoseBusters) can use the
    ``Mol`` directly — one bond-order assignment per copy serves both, rather
    than each caller independently re-extracting and re-assigning.

    Args:
        pdb_path (str): Path to the complex PDB file the ligand is in.
        smiles (str): Ground-truth SMILES for the ligand.
        ligand_resname (str): Residue name the ligand appears under in the
            PDB file. Defaults to ``"LIG"``.

    Returns:
        list[Chem.Mol]: One bonded, all-atom ``Mol`` per ligand copy that
            could be bond-order assigned. Empty if the file can't be parsed,
            has no ``ligand_resname`` residue, the SMILES can't be parsed, or
            every copy fails bond-order assignment — callers should fall back
            to their own ligand handling in that case. A copy that
            individually fails assignment (e.g. mismatched skeleton) is
            skipped, not treated as a whole-file failure.

    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None:
        logger.warning("RDKit could not parse {}", pdb_path)
        return []

    fragments = Chem.SplitMolByPDBResidues(mol)
    ligand = fragments.pop(ligand_resname, None)
    if ligand is None:
        logger.warning("No '{}' residue in {}", ligand_resname, pdb_path)
        return []

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        logger.warning("Could not parse SMILES {!r} for {}", smiles, pdb_path)
        return []

    copies = []
    for copy in Chem.GetMolFrags(ligand, asMols=True, sanitizeFrags=False):
        try:
            assigned = assign_bond_orders_from_template(template, copy)
        except Exception as exc:  # noqa: BLE001 - skip just this copy
            logger.warning(
                "Bond order assignment failed for one '{}' copy in {} ({})",
                ligand_resname,
                pdb_path,
                exc,
            )
            continue
        copies.append(assigned)
    return copies
