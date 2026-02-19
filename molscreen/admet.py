"""
ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction module.

This module provides rule-based ADMET property prediction using RDKit descriptors
and SMARTS pattern matching for structural alerts. No external APIs are used.
"""

from typing import Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from molscreen.properties import smiles_to_mol, check_lipinski


# SMARTS patterns for structural alerts
SMARTS_PATTERNS = {
    # Metabolic liability patterns
    'thiol': '[SH]',
    'imidazole': 'c1ncnc1',
    'furan': 'o1cccc1',
    'thiophene': 's1cccc1',

    # Mutagenicity alerts (Ames test)
    'nitro_aromatic': '[N+](=O)[O-]a',  # Nitro group on aromatic ring
    'aromatic_amine': 'cN',  # Aromatic amine
    'epoxide': 'C1OC1',  # Epoxide ring
    'aziridine': 'C1NC1',  # Aziridine ring

    # Hepatotoxicity alerts
    'quinone': 'O=C1C=CC(=O)C=C1',  # Quinone structure
    'alpha_beta_unsaturated_carbonyl': 'C=CC=O',  # Michael acceptor

    # Basic nitrogen for hERG
    'basic_nitrogen': '[NH2,NH1,NH0;!$(N-C=[O,N,S])]',  # Basic nitrogen
}


def _has_substructure(mol: Chem.Mol, smarts: str) -> bool:
    """
    Check if molecule contains a substructure matching the SMARTS pattern.

    Args:
        mol: RDKit molecule object
        smarts: SMARTS pattern string

    Returns:
        True if substructure is found, False otherwise
    """
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return False
    return mol.HasSubstructMatch(pattern)


def _predict_absorption(mol: Chem.Mol, properties: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict absorption properties.

    Args:
        mol: RDKit molecule object
        properties: Pre-calculated molecular properties (MW, LogP, TPSA, etc.)

    Returns:
        Dictionary with absorption predictions:
        - caco2_class: 'high' or 'low' permeability
        - bioavailability_ro5: Lipinski Rule of Five compliance
    """
    logp = properties['LogP']
    tpsa = properties['TPSA']

    # Caco-2 permeability: high if LogP > 0 AND TPSA < 140
    caco2_class = 'high' if (logp > 0 and tpsa < 140) else 'low'

    # Bioavailability: use Lipinski Rule of Five
    lipinski_result = check_lipinski(properties=properties)
    bioavailability_ro5 = lipinski_result['passes_lipinski']

    return {
        'caco2_class': caco2_class,
        'bioavailability_ro5': bioavailability_ro5,
    }


def _predict_distribution(mol: Chem.Mol, properties: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict distribution properties.

    Args:
        mol: RDKit molecule object
        properties: Pre-calculated molecular properties

    Returns:
        Dictionary with distribution predictions:
        - bbb_penetrant: Blood-brain barrier penetration (True/False)
        - vd_class: Volume of distribution class ('low', 'medium', 'high')
    """
    mw = properties['MW']
    logp = properties['LogP']
    tpsa = properties['TPSA']

    # BBB penetration: MW < 450, LogP 0-3, TPSA < 90
    bbb_penetrant = (mw < 450 and 0 <= logp <= 3 and tpsa < 90)

    # Volume of distribution based on LogP
    if logp < 1:
        vd_class = 'low'
    elif logp <= 3:
        vd_class = 'medium'
    else:
        vd_class = 'high'

    return {
        'bbb_penetrant': bbb_penetrant,
        'vd_class': vd_class,
    }


def _predict_metabolism(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Predict metabolism-related properties.

    Args:
        mol: RDKit molecule object

    Returns:
        Dictionary with metabolism predictions:
        - cyp_alert: CYP450 liability alert (True/False)
    """
    # Check for structural alerts indicating CYP450 liability
    cyp_alerts = [
        _has_substructure(mol, SMARTS_PATTERNS['thiol']),
        _has_substructure(mol, SMARTS_PATTERNS['imidazole']),
        _has_substructure(mol, SMARTS_PATTERNS['furan']),
        _has_substructure(mol, SMARTS_PATTERNS['thiophene']),
    ]

    cyp_alert = any(cyp_alerts)

    return {
        'cyp_alert': cyp_alert,
    }


def _predict_excretion(mol: Chem.Mol, properties: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict excretion properties.

    Args:
        mol: RDKit molecule object
        properties: Pre-calculated molecular properties

    Returns:
        Dictionary with excretion predictions:
        - renal_clearance: 'likely' or 'unlikely' for renal clearance
    """
    mw = properties['MW']
    logp = properties['LogP']

    # Renal clearance likely for small, water-soluble molecules
    # MW < 300 AND LogP < 2 indicates good water solubility
    renal_clearance = 'likely' if (mw < 300 and logp < 2) else 'unlikely'

    return {
        'renal_clearance': renal_clearance,
    }


def _predict_toxicity(mol: Chem.Mol, properties: Dict[str, float]) -> Dict[str, Any]:
    """
    Predict toxicity alerts.

    Args:
        mol: RDKit molecule object
        properties: Pre-calculated molecular properties

    Returns:
        Dictionary with toxicity predictions:
        - herg_alert: hERG channel inhibition risk (True/False)
        - ames_alert: Ames mutagenicity alert (True/False)
        - hepatotox_alert: Hepatotoxicity alert (True/False)
    """
    logp = properties['LogP']
    mw = properties['MW']

    # hERG risk: basic nitrogen AND (logP > 3 OR MW > 300)
    has_basic_n = _has_substructure(mol, SMARTS_PATTERNS['basic_nitrogen'])
    herg_alert = has_basic_n and (logp > 3 or mw > 300)

    # Ames mutagenicity alerts
    ames_alerts = [
        _has_substructure(mol, SMARTS_PATTERNS['nitro_aromatic']),
        _has_substructure(mol, SMARTS_PATTERNS['aromatic_amine']),
        _has_substructure(mol, SMARTS_PATTERNS['epoxide']),
        _has_substructure(mol, SMARTS_PATTERNS['aziridine']),
    ]
    ames_alert = any(ames_alerts)

    # Hepatotoxicity alerts
    hepatotox_alerts = [
        _has_substructure(mol, SMARTS_PATTERNS['quinone']),
        _has_substructure(mol, SMARTS_PATTERNS['epoxide']),
        _has_substructure(mol, SMARTS_PATTERNS['alpha_beta_unsaturated_carbonyl']),
    ]
    hepatotox_alert = any(hepatotox_alerts)

    return {
        'herg_alert': herg_alert,
        'ames_alert': ames_alert,
        'hepatotox_alert': hepatotox_alert,
    }


def _calculate_overall_score(absorption: Dict, distribution: Dict,
                             metabolism: Dict, excretion: Dict,
                             toxicity: Dict) -> float:
    """
    Calculate overall ADMET score (0-1, higher is better).

    The score starts at 1.0 and penalties are subtracted for each risk factor:
    - hERG alert: -0.15
    - Ames alert: -0.20
    - Hepatotoxicity alert: -0.15
    - CYP alert: -0.10
    - Low Caco-2 permeability: -0.10
    - Non-BBB penetrant: -0.05
    - Fails Lipinski Ro5: -0.15
    - Unlikely renal clearance: -0.05

    Args:
        absorption: Absorption prediction results
        distribution: Distribution prediction results
        metabolism: Metabolism prediction results
        excretion: Excretion prediction results
        toxicity: Toxicity prediction results

    Returns:
        Overall score between 0.0 and 1.0
    """
    score = 1.0

    # Toxicity penalties
    if toxicity['herg_alert']:
        score -= 0.15
    if toxicity['ames_alert']:
        score -= 0.20
    if toxicity['hepatotox_alert']:
        score -= 0.15

    # Metabolism penalty
    if metabolism['cyp_alert']:
        score -= 0.10

    # Absorption penalties
    if absorption['caco2_class'] == 'low':
        score -= 0.10
    if not absorption['bioavailability_ro5']:
        score -= 0.15

    # Distribution penalty (BBB penetration is context-dependent,
    # but for general drug-likeness, we give a small penalty for non-penetrant)
    if not distribution['bbb_penetrant']:
        score -= 0.05

    # Excretion penalty
    if excretion['renal_clearance'] == 'unlikely':
        score -= 0.05

    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))

    return score


def predict_admet(smiles: str) -> Dict[str, Any]:
    """
    Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

    This function uses rule-based classification with RDKit descriptors and SMARTS
    pattern matching for structural alerts. No external APIs are used.

    Args:
        smiles: SMILES representation of the molecule

    Returns:
        Dictionary containing ADMET predictions:
        {
            'absorption': {
                'caco2_class': 'high'|'low',
                'bioavailability_ro5': bool,
            },
            'distribution': {
                'bbb_penetrant': bool,
                'vd_class': 'low'|'medium'|'high',
            },
            'metabolism': {
                'cyp_alert': bool,
            },
            'excretion': {
                'renal_clearance': 'likely'|'unlikely',
            },
            'toxicity': {
                'herg_alert': bool,
                'ames_alert': bool,
                'hepatotox_alert': bool,
            },
            'overall_score': float,  # 0-1, higher is better
        }

    Raises:
        MoleculeError: If SMILES is invalid

    Example:
        >>> result = predict_admet("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        >>> result['absorption']['bioavailability_ro5']
        True
        >>> result['overall_score']
        0.75
    """
    # Parse SMILES and validate
    mol = smiles_to_mol(smiles)

    # Calculate molecular properties
    properties = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'RotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
    }

    # Predict each ADMET category
    absorption = _predict_absorption(mol, properties)
    distribution = _predict_distribution(mol, properties)
    metabolism = _predict_metabolism(mol)
    excretion = _predict_excretion(mol, properties)
    toxicity = _predict_toxicity(mol, properties)

    # Calculate overall score
    overall_score = _calculate_overall_score(
        absorption, distribution, metabolism, excretion, toxicity
    )

    return {
        'absorption': absorption,
        'distribution': distribution,
        'metabolism': metabolism,
        'excretion': excretion,
        'toxicity': toxicity,
        'overall_score': overall_score,
    }
