"""mcmc-robust: NUTS with calibrated coordinate-wise arcsinh transforms for heavy-tailed targets.

The main entry points are `sample_blackjax_unified_nuts` (JAX/BlackJAX) and `UnifiedRobustNUTS`
(NumPy). `GentleMALA`, `UnifiedRobustSampler` and `UnifiedRobustHMC` are experimental; see
REPORT.md for what each component does and does not establish.
"""

from mcmc_robust.integrations import (
    BlackJAXAlgorithm,
    JaxAffineTransform,
    JaxArcsinhTransform,
    PyMCModelTarget,
    as_blackjax_algorithm,
    sample_pymc_model,
)
from mcmc_robust.jax_unified_nuts import (
    JaxUnifiedNUTSResult,
    arviz_from_samples,
    calibrate_jax_affine_transform,
    calibrate_jax_arcsinh_transform,
    sample_blackjax_unified_nuts,
)
from mcmc_robust.samplers.automala import AutoMALA
from mcmc_robust.samplers.novel.gentle_mala import GentleMALA
from mcmc_robust.samplers.novel.nuts import ArcsinhTransformedNUTS, UnifiedRobustNUTS
from mcmc_robust.samplers.novel.unified import UnifiedRobustSampler
from mcmc_robust.samplers.novel.unified_hmc import UnifiedRobustHMC
from mcmc_robust.targets.roughness import NealsFunnelTarget
from mcmc_robust.targets.suite import get_target
from mcmc_robust.transforms.coordinate_arcsinh import CoordinateWiseArcsinhTransform

__version__ = "0.1.0"

__all__ = [
    "ArcsinhTransformedNUTS",
    "AutoMALA",
    "BlackJAXAlgorithm",
    "CoordinateWiseArcsinhTransform",
    "GentleMALA",
    "JaxAffineTransform",
    "JaxArcsinhTransform",
    "JaxUnifiedNUTSResult",
    "NealsFunnelTarget",
    "PyMCModelTarget",
    "UnifiedRobustHMC",
    "UnifiedRobustNUTS",
    "UnifiedRobustSampler",
    "arviz_from_samples",
    "as_blackjax_algorithm",
    "calibrate_jax_affine_transform",
    "calibrate_jax_arcsinh_transform",
    "get_target",
    "sample_blackjax_unified_nuts",
    "sample_pymc_model",
]
