"""Why the composite is block-structured, not sequential: tail curvature of a sequential composite.

Sequential composite (applied to *all* coordinates): z -> y = c sinh(z) (coordinate-wise) -> w = R sinhc(|y|) y
(radial) -> x = w (target already whitened). For an isotropic multivariate t (nu, d) the pullback along an
axis, z = t e_1, behaves like -(nu c / 2) e^t: the radial map has already made the tail exponential, and the
coordinate-wise map on top makes it doubly exponential (stiff, beta = infinity in the Livingstone et al. 2019
classification). The same happens in the other order. This script evaluates log pi_z and its second
derivative along the axis with JAX and compares with the radial-only and coordinate-only pullbacks.

Run: uv run python -m explore.tail_direction.sequential_composite_check
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .maps import Block, BlockMap, log_cosh_from_u, make_fns, pullback
from .targets import _mvt_logdensity

OUT = Path(__file__).parent / "results" / "sequential_composite_check.json"


def main() -> None:
    nu, d, big_r, c = 1.0, 10, 0.5, 1.0
    lp_x = _mvt_logdensity(nu, np.eye(d))
    radial = BlockMap(
        d, [Block("radial", tuple(range(d)), {"mu": np.zeros(d), "L": np.eye(d), "R": np.asarray(big_r)})]
    )
    coord = BlockMap(d, [Block("coord", tuple(range(d)), {"mu": np.zeros(d), "R": np.full(d, c)})])
    lp_rad = pullback(lp_x, radial.structure(), radial.param_tree())
    lp_crd = pullback(lp_x, coord.structure(), coord.param_tree())
    inv_rad, _, ld_rad = make_fns(radial.structure(), radial.param_tree())

    def lp_seq(z: jnp.ndarray) -> jnp.ndarray:
        y = c * jnp.sinh(z)
        ld_c = jnp.sum(jnp.log(c) + log_cosh_from_u(z * z))
        return lp_x(inv_rad(y)) + ld_rad(y) + ld_c

    rows = []
    e1 = jnp.zeros(d).at[0].set(1.0)
    for name, f in [("coord", lp_crd), ("radial", lp_rad), ("sequential_radial_then_coord", lp_seq)]:

        def along(t: jnp.ndarray, f=f) -> jnp.ndarray:
            return f(t * e1)

        d2 = jax.grad(jax.grad(along))
        # t <= 5 keeps |x|^2 < 1e300 so second derivatives of log1p(|x|^2 / nu) do not underflow
        rows.extend(
            {
                "map": name,
                "t": t,
                "log_density": float(along(jnp.asarray(t))),
                "d2_log_density": float(d2(jnp.asarray(t))),
            }
            for t in [1.0, 2.0, 3.0, 4.0, 5.0]
        )
    rows.extend(
        {"map": "prediction_sequential", "t": t, "d2_log_density": float(-nu * c * np.sinh(t))} for t in [3.0, 4.0, 5.0]
    )
    OUT.write_text(json.dumps({"nu": nu, "d": d, "R": big_r, "c": c, "rows": rows}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
