"""SABR model parameters, shared by every pricing method."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SABRParams:
    """SABR model parameters (absorbing boundary, standard parametrisation).

    We work with the log-volatility state Yt = log Y, so the initial state is
    (x0, log(y0)).
    """

    beta: float
    rho: float
    nu: float
    x0: float = 1.0  # initial forward / spot
    y0: float = 0.2  # initial volatility (not its log)

    def validate(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.nu <= 0.0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.x0 <= 0.0:
            raise ValueError(f"x0 must be > 0, got {self.x0}")
        if self.y0 <= 0.0:
            raise ValueError(f"y0 must be > 0, got {self.y0}")
        # Coercivity condition (Remark 2.18 of Horvath-Reichmann): |rho| * nu^2 < 2
        if abs(self.rho) * self.nu * self.nu >= 2.0:
            raise ValueError(
                f"Coercivity condition |rho|*nu^2 < 2 violated: "
                f"|{self.rho}|*{self.nu}^2 = {abs(self.rho) * self.nu ** 2}"
            )
