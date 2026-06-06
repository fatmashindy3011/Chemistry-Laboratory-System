import pytest
from omar_laws import BasicLaws, Solutions, Thermo

def test_moles_calculation():
    logic = BasicLaws()
    assert logic.moles(10, 2) == 5.0

def test_molarity_logic():
    sol = Solutions()
    assert sol.molarity(2, 1) == 2.0
    
def test_temp_conversion():
    th = Thermo()
    assert th.temp_c_to_k(0) == 273
    assert th.temp_k_to_c(273) == 0

def test_invalid_moles():
    logic = BasicLaws()
    with pytest.raises(ValueError):
        logic.moles(10, 5) 