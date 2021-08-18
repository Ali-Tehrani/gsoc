
r"""
Testing pyscf versus gbasis
"""
import iodata
import numpy as np
import pyscf.gto.basis.parse_nwchem
from iodata import load_one
from iodata.basis import convert_convention_shell
from gbasis.wrappers import from_iodata
from gbasis.evals.eval import evaluate_basis
import h5py
h5py.get_config().default_file_mode = 'r'
from pyscf import gto


# Maps atomic charge to atomic symbol for pyscf
charge_to_atom = {
    1: "H", 2: "He", 3 : "Li", 4 : "Be", 5: "B", 6 : "C", 7 : "N", 8: "O", 9 : "F",
    10 : "Ne", 11 : "Na", 12 : "Mg", 13 : "Al", 14 : "Si", 15 : "P", 16 : "S", 17 : "Cl", 18 : "Ar",
    19 : "K", 20 : "Ca", 21 : "Sc", 22 : "Ti", 23 : "V", 24 : "Cr", 25 : "Mn", 26 : "Fe",
    27 : "Co", 28 : "Ni", 29 : "Cu", 30 : "Zn", 31 : "Ga", 32 : "Ge", 33 : "As", 34 : "Se",
    35 : "Br", 36 : "Kr", 37 : "Rb", 38 : "Sr", 39 : "Y", 40 : "Zr", 41 : "Nb", 42 : "Mo",
    43 : "Tc", 44 : "Ru", 45 : "Rh", 46 : "Pd", 47 : "Ag", 48 : "Cd", 49 : "In", 50 : "Sn",
    51 : "Sb", 52 : "Te", 53 : "I", 54 : "Xe"}


def convert_iodata_obj_to_pyscf_gto(iodata_obj, cart=False):
    r"""
    Convert an iodata obj to PyScf object.

    Parameters
    ----------
    iodata_obj : iodata.IOData
        IOdata object
    cart : bool
        If True, then all shells are Cartesian, else it is Spherical.

    Returns
    -------
    (pyscf.gto.Mole, list)
        Returns a PyScf Mole object representing the basis set of the molecule.
        List[int, int, int] where first integer is the angular momentum and the
        second integer is the index where the order needs to be changed and
        the third integer is the final index where the order needs to be changed.

    """
    assert isinstance(iodata_obj, iodata.IOData)
    mol_pyscf = gto.M()
    natoms_mol = {}  # Stores number of times an atom shows up in molecule, keys are atom, items int.
    basis = {}  # Basis set is stored here. keys are pyscf atom symbol items are basis set.
    basis_atom_index = {}  # keys are atom index, items are pyscf atom symbol

    # Coordinates, charges and number of Atom
    coordinates = iodata_obj.atcoords
    charges = iodata_obj.atnums
    natoms = iodata_obj.natom

    # Update atomic string for pyscf that stores the atomic coordinates
    # Update the `natoms_mol` number of times a atom show sup
    # Create unique symbols for each atom to be identifiable.
    atom = ""
    for i in range(natoms):
        atom_symbol = charge_to_atom[charges[i]]

        # Update the number of times an atom shows up in `natoms`
        if atom_symbol not in natoms_mol:
            natoms_mol[atom_symbol] = 1
        else:
            natoms_mol[atom_symbol] += 1

        # Atom + ith Number, for example H2O would be "O", "H", "H2".
        if natoms_mol[atom_symbol] == 1:
            # If this is first time the atom shows up, then just use the atom symbol.
            basis_symbol_index = charge_to_atom[charges[i]]
        elif natoms_mol[atom_symbol] > 1:
            # If atom shows up more than once, add a number to index it e.g. O2, H2, H3
            basis_symbol_index = charge_to_atom[charges[i]] + \
                                 str(natoms_mol[charge_to_atom[charges[i]]])

        # Construct atom string storing the atomic coordinates of the atoms in the molecule.
        atom += basis_symbol_index + " "
        basis[basis_symbol_index] = []
        basis_atom_index[i] = basis_symbol_index
        for j in range(3):
            atom += str(coordinates[i, j])
            if j < 2:
                atom += " "
        if i < natoms - 1:
            atom += "; "
    mol_pyscf.atom = atom

    # Update basis_set for pysch
    shells = iodata_obj.obasis.shells
    where_spherical_starts = []
    basis_func_counter = 0
    # Go Through Each Contracted Shell
    for shell in shells:
        # Grab information about the shell
        icenter = shell.icenter
        angmoms = shell.angmoms
        exponents = shell.exponents
        coeffs = shell.coeffs
        kinds = shell.kinds

        # Go Through Each Segmented Shell in Contracted Shell
        for j, angmom in enumerate(angmoms):
            # If cartesian is False, then all angmom momentum >=2 should be spherical
            if not cart:
                if angmom >= 2:
                    # Assert that the kind shoulpd be Spherical
                    assert kinds[j] == "p"
            if cart:
                if angmom >= 2:
                    # Assert that the kind should be Cartesian
                    assert kinds[j] == "c"

            # Store the [angmom, (exponents, coefficients)]  inside `basis_seg_shell`
            basis_seg_shell = [int(angmom)]
            for i in range(len(exponents)):
                ith_exp = exponents[i]
                ith_jth_coeffs = coeffs[i, j]
                basis_seg_shell.append((ith_exp, ith_jth_coeffs))
            # Append the basis set to the unique atom symbol.
            basis[basis_atom_index[icenter]].append(
                basis_seg_shell
            )

            # Update basis func index
            #   Got the basis function numbers from https://www.chemissian.com/ch5
            #   Track where you have to change the order
            if angmom == 0:
                basis_func_counter += 1
            elif angmom == 1:
                basis_func_counter += 3
            elif angmom == 2:
                # If Cartesian Else spherical
                if kinds[j] == "c":
                    basis_func_counter += 6
                else:
                    basis_func_counter += 5
                    where_spherical_starts.append(
                        [angmom, basis_func_counter - 5, basis_func_counter]
                    )
            elif angmom == 3:
                # F-type orbitals
                if kinds[j] == "c":
                    basis_func_counter += 10
                else:
                    basis_func_counter += 7
                    where_spherical_starts.append(
                        [angmom, basis_func_counter - 7, basis_func_counter]
                    )
            else:
                raise NotImplementedError("Angular momentum greater than 3 isn't supported here.")
    mol_pyscf.basis = basis
    mol_pyscf.build(unit="B", cart=cart)
    return mol_pyscf, where_spherical_starts


def _get_permutation_from_pyscf_to_iodata_spherical(angmom, iodata_obj):
    r"""
    Returns the permutation from PyScf Spherical Order to IOdata order.

    For example:
    D-type orbitals
    # PyScf order: ["s2", "s1", "c0", "c1", "c2"]
    # IOdata order: ['c0', 'c1', 's1', 'c2', 's2'

    F-type orbitals
    # PyScf order: ["s3", "s2", "s1", "c0", "c1", "c2", "c3"]
    # IOdata order: ['c0', 'c1', 's1', 'c2', 's2', 'c3', 's3']

    """
    assert angmom >= 2, "Angular momentum should be greater than equal to two."
    kinds_iodata = iodata_obj.obasis.conventions[(angmom, "p")]
    # Generate the type of spherical function/kinds for pyscf
    kinds_pyscf = []
    for i in range(-angmom, angmom + 1):
        if i < 0:
            kinds_pyscf.append("s" + str(np.abs(i)))
        if i >= 0:
            kinds_pyscf.append("c" + str(np.abs(i)))
    # Use iodata to get the permutation from PySCF to IOData
    perm_iodata_to_pyscf, _ = convert_convention_shell(
        kinds_pyscf, kinds_iodata,
    )
    return np.array(perm_iodata_to_pyscf)


iodata_obj = load_one("./data/water_b3lyp_sto3g.fchk")
contr, type = from_iodata(iodata_obj)
grid_3d = np.random.random((100,3))
basis_gbasis = evaluate_basis(contr, grid_3d, coord_type=type).T
mol_pyscf, type_changes = convert_iodata_obj_to_pyscf_gto(iodata_obj)
# PYSCF: Evaluate
ao = mol_pyscf.eval_gto('GTOval_sph', grid_3d)

# Change the order for Spherical functions (d-type, f-type) from PyScf order to IOdata order.
for i in range(len(type_changes)):
    angmom, start_index, end_index = type_changes[i]
    perm = _get_permutation_from_pyscf_to_iodata_spherical(angmom, iodata_obj)
    ao[:, start_index:end_index] = ao[:, perm + start_index]

assert ao.shape == basis_gbasis.shape
assert np.all(np.abs(ao - basis_gbasis) < 1e-8)
