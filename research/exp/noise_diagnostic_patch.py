"""Diagnostic-only amendment for compact cell-year abundance innovations."""
OLD='''    u = np.asarray(sim["process_noise_u"], dtype=np.float64) * np.asarray(data["pn_active"], dtype=np.float64)[:, None]
    if "pn_P" in data:
        u = np.asarray(data["pn_P"], dtype=np.float64) @ u
    return float(_PN.get("sigma", 0.05)) * u @ np.asarray(data["pn_S"], dtype=np.float64).T'''
NEW='''    raw = np.asarray(sim["process_noise_u"], dtype=np.float64)
    active = np.asarray(data["pn_active"], dtype=np.float64)
    if _PN.get('form') == 'cell_iid':
        from src.model.minor_flexibility import iid_log_innovations
        return iid_log_innovations(raw, np.asarray(data['pn_time_indices']), len(active), float(_PN['sigma']))
    if _PN.get('form') == 'regional' and _PN.get('unit_cell_variance'):
        from src.model.minor_flexibility import regional_log_innovations
        return regional_log_innovations(raw, active, np.asarray(data['pn_S']), float(_PN['sigma']))
    u = raw * active[:, None]
    if "pn_P" in data:
        u = np.asarray(data["pn_P"], dtype=np.float64) @ u
    return float(_PN.get("sigma", 0.05)) * u @ np.asarray(data["pn_S"], dtype=np.float64).T'''
OLD_ACTIVE='''    u = np.asarray(sim["process_noise_u"])[np.asarray(data["pn_active"]) > 0]'''
NEW_ACTIVE='''    u = np.asarray(sim["process_noise_u"])
    if u.shape[0] == len(data['pn_active']):
        u = u[np.asarray(data['pn_active']) > 0]
    elif 'pn_time_indices' not in data or u.shape[0] != len(data['pn_time_indices']):
        raise ValueError('process innovation time indices do not match diagnostic rows')'''


def patch(text):
    if text.count(OLD)!=1 or text.count(OLD_ACTIVE)!=1:raise ValueError('ambiguous diagnostic amendment')
    result=text.replace(OLD,NEW).replace(OLD_ACTIVE,NEW_ACTIVE)
    compile(result,'amended_map_diagnostics.py','exec')
    return result