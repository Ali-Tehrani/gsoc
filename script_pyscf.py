
r"""
Testing pyscf versus gbasis
"""

import numpy as np
from iodata import load_one
from gbasis.wrappers import from_iodata
from gbasis.parsers import parse_nwchem, make_contractions
from gbasis.evals.eval import evaluate_basis
import h5py
h5py.get_config().default_file_mode = 'r'
from pyscf import gto, scf


iodata = load_one("./water_b3lyp_sto3g.fchk")
contr, type = from_iodata(iodata)
# basis = parse_nwchem("cc-pcvtz.0.nw")
# contr = make_contractions(basis, ["O"], np.array([[0., 0., 0.]]))
grid_3d = np.random.random((5,3))
print([c.angmom_components_cart for c in contr])
# GBASSIS: Eval atomic orbitals
basis = evaluate_basis(contr, grid_3d, coord_type=type)
print(np.sort(np.ravel(basis)))

mol_pyscf = gto.M(
    atom = 'O 4.51644546E-03  3.34481526E-03 -2.36215767E-03; H 5.24663564E-01  1.67637605E+00 4.73943314E-01; H 1.14171584E+00 -4.42706141E-01 -1.34557949E+00;',
    basis = 'sto-3g'
)
print(mol_pyscf.basis)
print(mol_pyscf.cart)
print(mol_pyscf.cart_labels())
print(mol_pyscf)
# PYSCF: Evaluate
ao = mol_pyscf.eval_gto('GTOval_sph', grid_3d)
print(np.sort(np.ravel(ao)))

assert 1 == 0


# get the points

def pyscf_ao_calc():
    ao = mol_pyscf.eval_gto('GTOval_sph', grid_3d)
    return ao

pyscf_ao = pyscf_ao_calc()
print(pyscf_ao)


def gbasis_ao_calc():

    mo = mol_chemtools.compute_molecular_orbital(grid_3d)
    return mo
gbasis_ao = gbasis_ao_calc()

