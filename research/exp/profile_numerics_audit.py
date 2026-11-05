"""Check directional descent at a saved nonstationary fit; no optimization."""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer.util import log_density

from harness import Sandbox, ap
from src.model import hmc_common as hc
from local_map_coordinates import MapCoordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sb = Sandbox(os.environ["SBX_REFINED"])
    with np.load(args.params) as saved:
        sb.lat.update({k:saved[k] for k in saved.files})
    trace = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d,data=sb.lat),0)).get_trace(data=sb.data,prior_scale=1.)
    coords = MapCoordinates(trace,exclude=["dispersal_logit_intercept","dispersal_logit_slope"] if ap.VEIT_DEPARTURE else [])
    del trace
    arrays, static = hc.split_data(sb.data)
    model = hc.hide_deterministics(hc.array_model(ap.build_model_2d,static))
    def objective(x,data):
        return -log_density(model,(),{"arrays":data},coords.constrain(x))[0]
    vg = jax.jit(jax.value_and_grad(objective))
    x = coords.x0
    value, gradient = vg(x,arrays)
    norm = jnp.linalg.norm(gradient)
    direction = -gradient / norm
    rows = []
    for step in (1e-7,1e-6,1e-5,1e-4,1e-3,.01):
        plus, gp = vg(x+step*direction,arrays)
        minus, gm = vg(x-step*direction,arrays)
        rows.append({"step":step,"change_down_gradient":float(plus-value),
                     "change_up_gradient":float(minus-value),"linear_prediction":float(-step*norm),
                     "central_directional_derivative":float((plus-minus)/(2*step)),
                     "gradient_directional_derivative":float(jnp.dot(gradient,direction)),
                     "secant_directional_curvature":float(jnp.dot(gp-gm,direction)/(2*step))})
    result = {"objective":float(value),"gradient_l2":float(norm),"directions":rows,
              "devices":[str(d) for d in jax.devices()],"scope":"Numerical derivative audit at saved point, no fitting"}
    (args.out / "audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
