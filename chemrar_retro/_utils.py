from collections.abc import Collection
import copy
from pathlib import Path
import typing

from rdkit import Chem as rd

T = typing.TypeVar("T")


def copy_dataclass(obj: T) -> T:
    new_obj = copy.copy(obj)

    if hasattr(new_obj, "__dict__"):
        for attr, value in list(new_obj.__dict__.items()):
            if isinstance(value, Collection):
                new_obj.__dict__[attr] = copy.copy(value)

    return new_obj


def mols_from_sdf(sdf_path: Path) -> list[rd.Mol]:
    """Extract RDKit molecules list from sdf file.

    :param sdf_path: Path to sdf file with molecules
    :return: List of RDKit molecules
    """
    sdf_system = rd.ForwardSDMolSupplier(
        sdf_path.as_posix(),
        # sanitize=True,
        removeHs=False,
    )

    molecules = []
    for sdf_mol in sdf_system:
        molecule = rd.Mol(sdf_mol)
        molecule.SetProp("source", sdf_path.as_posix())
        molecules.append(molecule)
    return molecules
