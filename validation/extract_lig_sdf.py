"""
Extract LIG residues from CYP3A4 structure-track CIF files and write as SDF files.
SMILES for bond order assignment are read from cyp-challenge_structure_TEST_BLINDED.csv.

Approach (after Pat Walters, practicalcheminformatics.blogspot.com):
  - Use GEMMI to extract the LIG residue and write it as a PDB block
  - Use RDKit MolFromPDBBlock to parse it (handles connectivity natively)
  - Use AssignBondOrdersFromTemplate with the SMILES from the CSV

Usage:
    python extract_lig_sdf.py [--cif-dir DIR] [--out-dir DIR] [--csv CSV]
"""
import argparse
import csv
from pathlib import Path

import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem


HERE = Path(__file__).parent


def load_smiles_map(csv_path: Path) -> dict[str, str]:
    with open(csv_path) as f:
        return {row["Molecule_Name"]: row["SMILES"] for row in csv.DictReader(f)}


def extract_lig_pdb_block(cif_path: Path) -> str:
    """Extract first LIG residue from CIF and return it as a PDB-format string."""
    st = gemmi.read_structure(str(cif_path))

    # Build a new single-residue structure for writing
    new_st = gemmi.Structure()
    new_st.name = st.name
    new_model = gemmi.Model("1")

    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "LIG":
                    new_chain = gemmi.Chain(chain.name)
                    new_chain.add_residue(res)
                    new_model.add_chain(new_chain)
                    new_st.add_model(new_model)
                    return new_st.make_pdb_string()

    raise ValueError(f"No LIG residue in {cif_path.name}")


def mol_from_cif_lig(cif_path: Path, smiles: str) -> Chem.Mol:
    """Extract LIG from CIF and assign bond orders from SMILES template."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    pdb_block = extract_lig_pdb_block(cif_path)
    rd_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True)
    if rd_mol is None:
        raise ValueError("RDKit could not parse PDB block")

    return AllChem.AssignBondOrdersFromTemplate(template, rd_mol)


def extract_one(cif_path: Path, smiles: str, out_dir: Path) -> Path:
    mol = mol_from_cif_lig(cif_path, smiles)
    out_path = out_dir / f"{cif_path.stem}_LIG.sdf"
    w = Chem.SDWriter(str(out_path))
    w.write(mol)
    w.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-dir", default=str(HERE),
                        help="Directory containing CYP3A4 structure CIF files (default: script dir)")
    parser.add_argument("--out-dir", default=str(HERE / "ligands_sdf"),
                        help="Output directory for SDF files (default: ligands_sdf/)")
    parser.add_argument("--csv", default=str(HERE / "cyp-challenge_structure_TEST_BLINDED.csv"),
                        help="CSV with Molecule_Name,SMILES columns")
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv)

    smiles_map = load_smiles_map(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        print(f"No .cif files found in {cif_dir}")
        return

    ok, failed = 0, []
    for cif in cif_files:
        name = cif.stem
        if name not in smiles_map:
            print(f"  SKIP {cif.name}: no SMILES in CSV")
            failed.append(name)
            continue
        try:
            out = extract_one(cif, smiles_map[name], out_dir)
            print(f"  OK   {cif.name} -> {out.name}")
            ok += 1
        except Exception as e:
            print(f"  ERR  {cif.name}: {e}")
            failed.append(name)

    print(f"\n{ok}/{len(cif_files)} succeeded", end="")
    if failed:
        print(f", {len(failed)} failed: {failed}")
    else:
        print()


if __name__ == "__main__":
    main()
