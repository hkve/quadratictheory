import numpy as np

def disassociate_2dof(atom1, atom2, distances):
    return [
        f"{atom1} 0 0 0; {atom2} 0 0 {distance}" for distance in distances
    ]

def disassociate_h2o(distances):
    distances = distances if type(distances) in [list, tuple, np.ndarray] else [distances]
    theta = np.deg2rad(109.5)
    alpha = theta/2

    geometries = []
    for distance in distances:
        x, y = distance*np.cos(alpha), distance*np.sin(alpha)
        geometry = f"O 0 0 0; H {x} {y} 0; H {x} {-y} 0;"
        geometries.append(geometry)

    return geometries

LiH_ccpVDZ = {
    "R": [3.050, 6.100, 9.150],
    "geometry": [
        "Li 0 0 0; H 0 0 3.050",
        "Li 0 0 0; H 0 0 6.100",
        "Li 0 0 0; H 0 0 9.150",
    ]
}

N2_sto3g = {
    "R": [2.362, 2.836, 3.308],
    "geometry": [
        "N 0 0 0; N 0 0 2.362",
        "N 0 0 0; N 0 0 2.836",
        "N 0 0 0; N 0 0 3.308",
    ]
}

N2_ccpVTZ = {
    "R": [3.308, 3.608, 3.958],
    "geometry": [
        "N 0 0 0; N 0 0 3.308",
        "N 0 0 0; N 0 0 3.608",
        "N 0 0 0; N 0 0 3.958",
    ]
}
"""
[1]: Index of multi-determinantal and multi-reference character in coupled-cluster theory: https://doi.org/10.1063/5.0029339
"""
# All bond lengths are in Ångstrøm!
SR_examples = dict(
    He=f"He 0.0 0.0 0.0",
    Ne=f"Ne 0.0 0.0 0.0",
    Ar=f"Ar 0.0 0.0 0.0",
    H2=f"H 0.0 0.0 0.0; H 0.0 0.0 0.7414",
    O2=f"O 0.0 0.0 0.0; O 0.0 0.0 1.2075",
    N2=f"N 0.0 0.0 0.0; N 0.0 0.0 1.0977",
    F2=f"F 0.0 0.0 0.0; F 0.0 0.0 1.4119",
    HF=f"H 0.0 0.0 0.0; F 0.0 0.0 0.9168",
    LiF=f"Li 0.0 0.0 0.0; F 0.0 0.0 1.5639",
    BeH2=f"""
	Be 
	H 1 1.3264
	H 1 1.3264 2 180.0
	""",
    H2O="""
	O 
	H 1 0.9578
	H 1 0.9578 2 104.478
	""",
    H4=f"H 0.0 0.0 -2.11671; H 0.0 0.0 -1.05835; H 0.0 0.0 0.0; H 0.0 0.0 1.05835",
    BH3=f"""
    B
    H 1 1.19
    H 1 1.19 2 120
    H 1 1.19 2 120 3 180
    """,
    CH4=f"""
    C
    H 1 1.087
    H 1 1.087 2 109.471
    H 1 1.087 2 109.471 3 120
    H 1 1.087 2 109.471 3 -120
    """,
    C2H4=f"""
    C
    C 1 1.339
    H 1 1.086 2 121.2
    H 1 1.086 2 121.2 3 180
    H 2 1.086 1 121.2 3 0
    H 2 1.086 1 121.2 4 0 
    """,
)

MR_examples = dict(
    Ne_2p=f"Ne 0.0 0.0 0.0",
    H2=f"H 0.0 0.0 0.0; H 0.0 0.0 5.0",
    Li2=f"Li 0.0 0.0 0.0; Li 0.0 0.0 8.0",
    C2=f"C 0.0 0.0 0.0; C 0.0 0.0 5.0",
    N2=f"N 0.0 0.0 0.0; N 0.0 0.0 5.0",
    F2=f"F 0.0 0.0 0.0; F 0.0 0.0 5.0",
    HF=f"H 0.0 0.0 0.0; F 0.0 0.0 4.0",
    BN=f"B 0.0 0.0 0.0; N 0.0 0.0 1.2740",
    MgO=f"Mg 0.0 0.0 0.0; O 0.0 0.0 1.7679",
    H2O="""
	O 
	H 1 1.9156
	H 1 1.9156 2 104.478
	""",
    BCm=f"B 0.0 0.0 0.0; C 0.0 0.0 1.3809",
    CNp=f"C 0.0 0.0 0.0; N 0.0 0.0 1.1760",
    N2_2p=f"N 0.0 0.0 0.0; N 0.0 0.0 1.1291",
    BeNm=f"Be 0.0 0.0 0.0; N 0.0 0.0 1.4582",
)

"""
[2]: W4-11: A high-confidence benchmark dataset for computational thermochemistry derived from first-principles W4 data: 
https://doi.org/10.1016/j.cplett.2011.05.007
"""
W14_11_MR = dict(
    Be2=f"""
Be 
Be 1 2.50131878
""",
    B2=f"""
B
B 1 1.587553
""",
    BN=f"""
B
N 1 1.2830 
""",
    C2=f"""
C
C 1 1.24 
""",
    OF=f"""
O 
F 1 1.35300212 
""",
    O3=f"""
O
O 1 1.26881389
O 1 1.26881389 2 117.11828947
""",
    FO2=f"""
F
O 1 1.63240130
O 2 1.19195858 1 110.90510825
""",
    F2O=f"""
F
O 1 1.40558760
F 2 1.40558760 1 103.09788981
""",
    FOOF=f"""
F
O 1 1.53229741
O 2 1.23407466 1 108.45853898
F 3 1.53229741 2 108.45853898 1 87.52399646
""",
    ClOO=f"""
Cl
O 1 2.03230554
O 2 1.20810395 1 115.36876732
""",
    Cl2O=f"""
O
Cl 1 1.70054419
Cl 1 1.70054419 2 110.96534087
""",
    OClO=f"""
Cl
O 1 1.47287094
O 1 1.47287094 2 117.48606790
""",
    c_hooo=f"""
O  0.0    0.3034159717   -1.0937127481;
O  0.0   -0.5273609495   -0.1609698697;
O  0.0    0.1563085034    1.1992562818;
H  0.0    1.0736184779    0.8798024868
""",
    t_hooo=f"""
O
O 1 1.228
O 2 1.587 1 109.80
H 3 0.968 2 97.10 1 180.0
""",
    S3=f"""
S
S 1 1.9198308
S 1 1.9198308 2 117.3595
""",
    S4_c2v=f"""
S
S 1 1.90634338 
S 2 2.13701674 1 104.70263744 
S 3 1.90634338 2 104.70263744 1 0.0
""",
)
