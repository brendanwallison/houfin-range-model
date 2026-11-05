"""Fixed smooth crowding curvature shared by simulation and viability diagnostics."""
from src.config_utils import load_age_model_config

CROWDING_POWER = float((load_age_model_config()['population_model'].get('reproduction') or {}).get('crowding_power', 1.))
if CROWDING_POWER not in (1., 2.):
    raise ValueError('This fixed-curvature experiment supports crowding_power 1 or 2')


def crowding_argument(ratio, power=None):
    """Preserve array backend and the original power-one operation exactly."""
    power = CROWDING_POWER if power is None else power
    if power == 1.:
        return ratio
    if power == 2.:
        return ratio * ratio
    raise ValueError('crowding power must be 1 or 2')
