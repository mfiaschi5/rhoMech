"""Fixed set of small molecules used to test every featurization function."""

MOLECULES: dict[str, str] = {
    "ethane": "CC",
    "ethanol": "CCO",
    "acetone": "CC(=O)C",
    "acetaldehyde": "CC=O",
    "methyl_acetate": "CC(=O)OC",
    "chloromethane": "CCl",
    "methylamine": "CN",
    "acetonitrile": "CC#N",
    "benzene": "c1ccccc1",
    "acrolein": "C=CC=O",
}

# Open-shell fixtures (open-shell plan P0), written by hand, never copied from a dataset: SMILES
# and the radical electrons RDKit assigns as drawn. The drawings RDKit reads wrongly are on
# purpose (Cl[C]Cl and [C]=O are singlets, [N]=[N] is N2, ozone is a closed-shell singlet;
# RDKit gives the d-shell metals no unpaired electron): every featurization function changed by
# a later open-shell phase is tested on them, beside the closed-shell set above.
OPEN_SHELL: dict[str, tuple[str, int]] = {
    "methyl": ("[CH3]", 1),
    "ethyl": ("C[CH2]", 1),
    "allyl": ("C=C[CH2]", 1),
    "benzyl": ("[CH2]c1ccccc1", 1),
    "phenoxyl": ("[O]c1ccccc1", 1),
    "tempo": ("CC1(C)CCCC(C)(C)N1[O]", 1),
    "superoxide": ("[O][O-]", 1),
    "benzene_radical_anion": ("[CH-]1C=CC=C[CH]1", 1),
    "triplet_o2": ("[O][O]", 2),
    "bromine_atom": ("[Br]", 1),
    "hydroxyl": ("[OH]", 1),
    "nitric_oxide": ("[N]=O", 1),
    "dichlorocarbene": ("Cl[C]Cl", 2),
    "co_drawn": ("[C]=O", 2),
    "n2_drawn": ("[N]=[N]", 2),
    "ozone_drawn": ("[O]O[O]", 2),
    "ozone_charge_separated": ("[O][O+][O-]", 2),
    "copper_dichloride": ("Cl[Cu]Cl", 0),
    "tetrabromoferrate": ("Br[Fe-](Br)(Br)Br", 0),
}
