"""Energy / CO2 tracking (CodeCarbon) and a lightweight online cost model.

  * EnergyTracker: measure kWh / CO2 for any code block.
  * CostModel: predict per-config energy for CAFA from observed
    (backbone, resolution, epochs) -> energy samples
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from codecarbon import EmissionsTracker  # type: ignore
    _HAS_CODECARBON = True
except Exception:
    _HAS_CODECARBON = False


class EnergyTracker:
    """Context manager returning (energy_kwh, co2_kg, seconds).

    Falls back to a wall-clock-only estimate if codecarbon is unavailable.
    """

    def __init__(self, project_name: str = "greenvision", enabled: bool = True):
        self.enabled = enabled and _HAS_CODECARBON
        self.project_name = project_name
        self._tracker = None
        self._t0 = 0.0
        self.energy_kwh = 0.0
        self.co2_kg = 0.0
        self.seconds = 0.0
        self.measured = False   

    def __enter__(self) -> "EnergyTracker":
        self._t0 = time.time()
        if self.enabled:
            try:
                self._tracker = EmissionsTracker(
                    project_name=self.project_name,
                    log_level="error",
                    save_to_file=False,
                    tracking_mode="process",
                )
                self._tracker.start()
            except Exception as e:  
                logger.warning("CodeCarbon start failed (%s); using time only.", e)
                self.enabled = False
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.time() - self._t0
        if self.enabled and self._tracker is not None:
            try:
                self.co2_kg = float(self._tracker.stop() or 0.0)
                data = getattr(self._tracker, "final_emissions_data", None)
                if data is not None and getattr(data, "energy_consumed", None):
                    self.energy_kwh = float(data.energy_consumed)
                    self.measured = True
            except Exception as e:  
                logger.warning("CodeCarbon stop failed (%s).", e)
        if not self.measured:
            
            self.energy_kwh = (70.0 * self.seconds) / 3_600_000.0
            logger.warning(
                "Energy MODELLED at 70W for %.0fs -- CodeCarbon returned no "
                "measurement.", self.seconds)


class CostModel:
    """Predicts per-config energy (kWh) for the CAFA acquisition.

    Before observations exist it uses a static prior from backbone cost weights
    scaled by (resolution^2 * epochs). It then refines a per-backbone
    energy-per-(pixel*epoch) coefficient from real EnergyTracker samples.
    """

    def __init__(self, backbone_prior: Dict[str, float]):
        self.backbone_prior = backbone_prior
        self._coeff: Dict[str, float] = {}
        self._samples: Dict[str, list] = defaultdict(list)

    def observe(self, backbone: str, resolution: int, epochs: int, energy_kwh: float) -> None:
        denom = max(resolution * resolution * max(epochs, 1), 1)
        self._samples[backbone].append(energy_kwh / denom)
        s = sorted(self._samples[backbone])
        self._coeff[backbone] = s[len(s) // 2]

    def predict(self, backbone: str, resolution: int, epochs: int) -> float:
        denom = resolution * resolution * max(epochs, 1)
        if backbone in self._coeff:
            return self._coeff[backbone] * denom
        prior_w = self.backbone_prior.get(backbone, 1.0)
        return prior_w * denom * 1e-9

    def normalized_cost(self, backbone: str, resolution: int, epochs: int) -> float:
        """Cost in [~0,1] relative to the most expensive point in the space."""
        pred = self.predict(backbone, resolution, epochs)
        ref = max(
            [self.predict(bb, resolution, epochs) for bb in self.backbone_prior]
            + [pred]
        )
        return pred / max(ref, 1e-12)
