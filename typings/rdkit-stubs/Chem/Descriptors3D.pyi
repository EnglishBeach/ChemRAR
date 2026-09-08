"""
 Descriptors derived from a molecule's 3D structure

"""
from __future__ import annotations
from rdkit.Chem.Descriptors import _isCallable
from rdkit.Chem import rdMolDescriptors
__all__: list[str] = ['CalcMolDescriptors3D', 'descList', 'rdMolDescriptors']
def CalcMolDescriptors3D(mol, confId = -1):
    """
    
        Compute all 3D descriptors of a molecule
        
        Arguments:
        - mol: the molecule to work with
        - confId: conformer ID to work with. If not specified the default (-1) is used
        
        Return:
        
        dict
            A dictionary with decriptor names as keys and the descriptor values as values
    
        raises a ValueError 
            If the molecule does not have conformers
        
    """
def _setupDescriptors(namespace):
    ...
descList: list  # value = [('PMI1', <function <lambda> at 0x000001A4FBB97B50>), ('PMI2', <function <lambda> at 0x000001A4FC2C8550>), ('PMI3', <function <lambda> at 0x000001A4FC2C85E0>), ('NPR1', <function <lambda> at 0x000001A4FC2C8670>), ('NPR2', <function <lambda> at 0x000001A4FC2C8700>), ('RadiusOfGyration', <function <lambda> at 0x000001A4FC2C8790>), ('InertialShapeFactor', <function <lambda> at 0x000001A4FC2C8820>), ('Eccentricity', <function <lambda> at 0x000001A4FC2C88B0>), ('Asphericity', <function <lambda> at 0x000001A4FC2C8940>), ('SpherocityIndex', <function <lambda> at 0x000001A4FC2C89D0>), ('PBF', <function <lambda> at 0x000001A4FC2C8A60>)]
