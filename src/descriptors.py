"""RDKit descriptor calculation and drug-name resolution.

Everything the LLM is later told about the molecule originates here. If a number is
not computed in this module (or in `solubility_model`), the LLM has no business
asserting it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .schemas import MolecularDescriptors

RDLogger.DisableLog("rdApp.*")  # RDKit chatters on every failed parse

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class InvalidSmilesError(ValueError):
    """Raised when RDKit cannot parse the input as a molecule."""


class NameResolutionError(ValueError):
    """Raised when a drug name cannot be resolved to a structure."""


def resolve_name_to_smiles(name: str, timeout: int = 20, retries: int = 3) -> str:
    """Look a drug name up in PubChem and return its canonical SMILES.

    We resolve names against a real chemical database rather than asking the LLM,
    because an LLM recalling a SMILES string from memory is exactly the failure mode
    this project exists to avoid.

    PubChem throttles bursts (HTTP 503/429), so a naive single attempt fails
    intermittently for reasons that have nothing to do with the compound. Retry with
    backoff, and distinguish "no such compound" (404, terminal) from "try again".
    """
    slug = urllib.parse.quote(name.strip())
    # PubChem renamed this property to ConnectivitySMILES; older deployments still
    # serve CanonicalSMILES. Try each before giving up.
    last_error = "no response"

    for prop in ("ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES"):
        url = f"{PUBCHEM}/compound/name/{slug}/property/{prop}/JSON"
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                smiles = data["PropertyTable"]["Properties"][0].get(prop)
                if smiles:
                    return smiles
                last_error = f"{prop} absent from PubChem response"
                break
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code == 404:
                    break  # this property/name genuinely isn't there; try next property
                time.sleep(1.5 * (attempt + 1))  # throttled or server-side: back off
            except Exception as exc:  # noqa: BLE001 - network flake
                last_error = str(exc)
                time.sleep(1.5 * (attempt + 1))

    raise NameResolutionError(
        f"Could not resolve {name!r} in PubChem ({last_error}). "
        "Check the spelling or enter a SMILES string instead."
    )


def parse_molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSmilesError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def compute_descriptors(smiles: str) -> MolecularDescriptors:
    """Calculate the descriptor panel used for solubility risk and prompting."""
    mol = parse_molecule(smiles)

    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = Lipinski.NumRotatableBonds(mol)

    violations = sum([mw > 500, clogp > 5, hbd > 5, hba > 10])
    # Veber: oral bioavailability tends to require limited flexibility and polarity.
    veber = rotb <= 10 and tpsa <= 140

    return MolecularDescriptors(
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol),
        molecular_weight=round(mw, 2),
        clogp=round(clogp, 2),
        tpsa=round(tpsa, 2),
        h_bond_donors=hbd,
        h_bond_acceptors=hba,
        rotatable_bonds=rotb,
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        fraction_csp3=round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        molar_refractivity=round(Crippen.MolMR(mol), 2),
        lipinski_violations=violations,
        lipinski_pass=violations <= 1,
        veber_pass=veber,
    )


def resolve_input(name: str | None, smiles: str | None) -> tuple[str, str | None]:
    """Turn whatever the user typed into a SMILES string.

    Returns (smiles, resolved_name). An explicit SMILES always wins over a name, so
    the user can override a bad PubChem hit.
    """
    if smiles and smiles.strip():
        parse_molecule(smiles.strip())  # fail fast on garbage input
        return smiles.strip(), (name.strip() if name and name.strip() else None)
    if name and name.strip():
        return resolve_name_to_smiles(name), name.strip()
    raise ValueError("Provide either a drug name or a SMILES string.")
