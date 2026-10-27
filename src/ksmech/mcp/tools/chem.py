"""Chemistry tools: normalize reactants, a molecule's GFN2 properties, point group and frontier
orbitals, and the 2D depiction of a pool as SVG."""

from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ksmech.client import KsmechAPIError, KsmechClient
from ksmech.client import summary as sm
from ksmech.mcp.tools import READ, Enabled, add, blocking, dumps

Reactants = Annotated[
    list[str],
    Field(
        min_length=1,
        max_length=6,
        description="Reactant SMILES, one species per entry (charges explicit, e.g. 'C[O-]', "
        "'CCl'); a '.'-joined entry is one field holding several species.",
    ),
]
Stoichiometry = Annotated[
    list[int] | None,
    Field(description="Copies of each reactant (same length, each 1..4); default all 1."),
]
Smiles = Annotated[str, Field(description="One closed-shell molecule as SMILES, e.g. 'CC(=O)C'.")]
Detail = Annotated[bool, Field(description="Return the long form (more fields per row).")]


def _molecule_id(client: KsmechClient, smiles: str) -> str:
    """The id of a featurized molecule (created and waited for)."""
    mol = client.molecule_ready(smiles=smiles)
    if mol.status == "failed":
        err = mol.error
        raise KsmechAPIError(
            err.code if err else "FEATURIZE_FAILED", err.message if err else smiles
        )
    return mol.molecule_id


class ChemTools:
    """The chemistry tools over one client."""

    def __init__(self, client: KsmechClient) -> None:
        self.client = client

    async def normalize_reactants(
        self, reactants: Reactants, stoichiometry: Stoichiometry = None
    ) -> str:
        """Check and canonicalize reactant SMILES before a prediction, without computing
        anything. Returns ok, the pool key (sigma keys of the species), the state_id of the pool,
        the atom-mapped SMILES (its atom indices are the 'pool atoms' every step matrix
        uses) and one row per species (canonical key, charge, multiplicity, atoms, errors).
        An invalid input raises an error naming the field and the code (INVALID_SMILES,
        RADICAL_NOT_SUPPORTED, UNSUPPORTED_ELEMENT, LIMIT_EXCEEDED, ...)."""
        out = await blocking(self.client.normalize, reactants, stoichiometry)
        return dumps(sm.normalize_summary(out))

    async def molecule_properties(self, smiles: Smiles, detail: Detail = False) -> str:
        """GFN2-xTB properties of one molecule (computed on first use, then cached): HOMO,
        LUMO, gap, chemical potential mu, hardness eta and electrophilicity omega in eV, and
        per atom the formal and Mulliken charges, frozen-orbital Fukui f+ / f- (where a
        nucleophile / an electrophile attacks), the dual descriptor and %Vbur (steric
        burial). Atom indices are the record's (RDKit order with explicit H). detail adds
        the Lewis bonds with Wiberg indices and lengths, and diagnostic labels."""
        mid = await blocking(_molecule_id, self.client, smiles)
        props = await blocking(self.client.molecule_properties, mid)
        return dumps(sm.properties_summary(props, detail=detail))

    async def molecule_symmetry(
        self,
        smiles: Smiles,
        tolerance: Annotated[
            float | None, Field(ge=0.01, le=0.5, description="Angstrom (server default if unset).")
        ] = None,
        detail: Detail = False,
    ) -> str:
        """The Schoenflies point group of a molecule's GFN2 geometry (C1, Cs, C2v, D6h, Td,
        ...), its order, the kind of top, the RMS distance to the symmetrized structure, the
        sets of symmetry-equivalent atoms, and a higher group reachable at a looser
        tolerance. detail lists every symmetry element."""
        mid = await blocking(_molecule_id, self.client, smiles)
        pg = await blocking(self.client.molecule_symmetry, mid, tolerance=tolerance)
        return dumps(sm.symmetry_summary(pg, detail=detail))

    async def orbital_summary(
        self,
        smiles: Smiles,
        orbital_set: Annotated[
            str | None,
            Field(
                description="canonical (MOs with HOMO/LUMO labels), ibo or pm_mulliken "
                "(localized: lone pairs, bonds, antibonds); server default if unset."
            ),
        ] = None,
        window: Annotated[int, Field(ge=1, le=20, description="Orbitals per side.")] = 4,
        detail: Annotated[bool, Field(description="Every orbital, not only the window.")] = False,
    ) -> str:
        """The orbitals of a molecule around the frontier: the `window` highest occupied and
        lowest virtual ones, each with its energy (eV), occupation, label (HOMO, LUMO+1 for
        canonical MOs) and its four main atomic centres with their weights (e.g. 'O1:0.81').
        The 'type' field is a diagnostic label (lone pair, sigma, pi*...), never a model
        input."""
        mid = await blocking(_molecule_id, self.client, smiles)
        os_ = await blocking(self.client.molecule_orbitals, mid, orbital_set=orbital_set)
        return dumps(sm.orbitals_summary(os_, window=window, detail=detail))

    async def depict_svg(self, reactants: Reactants, stoichiometry: Stoichiometry = None) -> str:
        """A 2D drawing of a pool of reactants as SVG text (RDKit), with the pool's state_id
        (usable in check_step). Builds and saves the pool state, so the first call can take
        a while (its tokens are computed). Answers {"state_id", "mapped_smiles", "svg"}."""
        pool = await blocking(self.client.create_pool, reactants, stoichiometry)
        dep = await blocking(self.client.pool_depiction, pool.state_id, server_svg=True)
        out = {"state_id": pool.state_id, "mapped_smiles": pool.mapped_smiles}
        return dumps(out | {"svg": dep.server_svg})


def register(app: FastMCP, client: KsmechClient, enabled: Enabled) -> list[str]:
    t = ChemTools(client)
    names = [add(app, t.normalize_reactants, "Normalize reactants", READ)]
    names.append(add(app, t.depict_svg, "Depict a pool as SVG", READ))
    if enabled("workbench"):
        names.append(add(app, t.molecule_properties, "Molecule properties (GFN2)", READ))
        names.append(add(app, t.molecule_symmetry, "Molecule point group", READ))
        names.append(add(app, t.orbital_summary, "Frontier orbitals", READ))
    return names
