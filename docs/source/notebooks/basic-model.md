---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Basic demo

Here we give a basic demo of how to work with OpenSCM-Calibration.

## Imports

```{code-cell} ipython3
from functools import partial
from typing import Dict, Tuple, Any, Callable, Dict

import emcee
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import datetime as dt
import attr
from attrs import define, field
from openscm_units import unit_registry

import pandas as pd
import pint
import scipy.integrate
import scmdata.run
from emcwrap import DIMEMove
from multiprocess import Pool, Manager
from openscm_units import unit_registry as UREG
from tqdm.notebook import tqdm

from openscm_calibration import emcee_plotting
from openscm_calibration.cost import OptCostCalculatorSSE
from openscm_calibration.emcee_utils import (
    get_acceptance_fractions,
    get_autocorrelation_info,
)
from openscm_calibration.minimize import to_minimize_full
from openscm_calibration.model_runner import OptModelRunner
from openscm_calibration.scipy_plotting import (
    CallbackProxy,
    OptPlotter,
    get_ymax_default,
)
from openscm_calibration.scmdata_utils import scmrun_as_dict
from openscm_calibration.store import OptResStore
```

## Background

Solve model for emmisions and concentrations


$$\dfrac{d[CH_4]}{dt} = S_{CH_4} - k_1 [OH][CH_4]$$

$$\dfrac{d[H_2]}{dt} = S_{H_2} - k_2 [OH][H_2] - \frac{1}{\tau_{dep}} [H_2]$$

$$\dfrac{d[CO]}{dt} = S_{CO} - k_3 [OH][CO] + k_1 [OH][CH_4]$$

$$\dfrac{d[OH]}{dt} = S_{OH} - k_x [OH] - k_3 [OH][CO] - k_2 [OH][H_2] - k_1 [OH][CH_4]$$

+++

We are going to solve this system using [scipy's solve initial value problem](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

We want to support units, so we implement this using [pint](https://pint.readthedocs.io/). To make this work, we also have to define some wrappers. We define these in this notebook to show you the full details. If you find them distracting, please [raise an issue](https://github.com/openscm/OpenSCM-Calibration/issues/new) or submit a pull request.

+++

## Experiments

We're going to calibrate the model's response in two experiments:

- starting out of equilibrium
- out of equilibrium with a change in emissions

```{code-cell} ipython3
LENGTH_UNIT = "cm"
TIME_UNIT = "s"
CONC_UNIT = "ppb"
EMMS_UNIT = f"{CONC_UNIT} / {TIME_UNIT}"
RATE_CONSTANT_UNIT = f"1 / ({CONC_UNIT} {TIME_UNIT})"

# H2_UNIT = "H2"
# CH4_UNIT = "CH4"
# OH_UNIT = "OH"
# CO_UNIT = "CO"


UNIT_REGISTRY = unit_registry
# add hydrogen in here
symbol = "H2"
value = "hydrogen"
UNIT_REGISTRY.define(f"{symbol} = [{value}]")
UNIT_REGISTRY.define(f"{value} = {symbol}")
UNIT_REGISTRY._add_mass_emissions_joint_version(symbol)

symbol = "OH"
value = "hydroxyl"
UNIT_REGISTRY.define(f"{symbol} = [{value}]")
UNIT_REGISTRY.define(f"{value} = {symbol}")
UNIT_REGISTRY._add_mass_emissions_joint_version(symbol)

_EMMS_INDEXES = {
    "Emissions|CH4": 0,
    "Emissions|H2": 1,
    "Emissions|CO": 2,
    "Emissions|OH": 3,
}

_CONC_INDEXES = {k.split("|")[1].lower(): i for k, i in _EMMS_INDEXES.items()}
_CONC_INDEXES_INV = {v: k for k, v in _CONC_INDEXES.items()}
_CONC_ORDER = [_CONC_INDEXES_INV[i] for i in range(len(_CONC_INDEXES))]
```

```{code-cell} ipython3
def check_units(
    required_dimension: str,
) -> Callable[[Any, attr.Attribute, pint.Quantity[Any]], None]:
    """
    Check units of class attribute

    Intended to be used as a validator with :func:`attrs.field`

    Parameters
    ----------
    required_dimension
        Dimension that the input is required to have

    Returns
    -------
    Function that will validate that the intended dimension is passed
    """

    def check_unit_internal(  # pylint:disable=unused-argument
        self: Any, attribute: attr.Attribute, value: pint.Quantity[Any]
    ) -> None:
        """
        Check units of attribute

        Parameters
        ----------
        self
            Object instance

        attribute
            Attribute to check

        value
            Value to check

        Raises
        ------
        :obj:`pint.errors.DimensionalityError`
            Units are not the correct dimensionality
        """
        if not value.check(required_dimension):
            raise pint.errors.DimensionalityError(
                value,
                "a quantity with dimensionality",
                value.dimensionality,
                UNIT_REGISTRY.get_dimensionality(required_dimension),
                extra_msg=f" to set attribute `{attribute.name}`",
            )

    return check_unit_internal
```

```{code-cell} ipython3
def get_emms_func(scmrun):
    """
    Convert scmrun into a function which takes input time
    and returns emissions as a vector with the right units
    """
    # bad hard-coding, but can be fixed
    ts = scmrun.timeseries(time_axis="year")
    times = (
        UNIT_REGISTRY.Quantity(np.array(ts.columns), "year")
        .to(TIME_UNIT)
        .magnitude.squeeze()
    )
    emms = np.zeros_like(ts)
    for (v, u), df in ts.groupby(["variable", "unit"]):
        emms[_EMMS_INDEXES[v]] = (
            UNIT_REGISTRY.Quantity(df.values, u).to(EMMS_UNIT).magnitude
        )

    def emms_func(t):
        return emms[:, np.argmax(t <= times)]

    return emms_func
```

```{code-cell} ipython3
@define
class HydrogenBox:
    k1: pint.Quantity[float] = field(
        validator=check_units(f"{LENGTH_UNIT}^3 / {TIME_UNIT}")
    )
    """Rate constant for reaction of methane and hydroxyl radical [cm^3 / s]"""

    k2: pint.Quantity[float] = field(
        validator=check_units(f"{LENGTH_UNIT}^3 / {TIME_UNIT}")
    )
    """Rate constant for reaction of hydrogen and hydroxyl radical [cm^3 / s]"""

    k3: pint.Quantity[float] = field(
        validator=check_units(f"{LENGTH_UNIT}^3 / {TIME_UNIT}")
    )
    """Rate constant for reaction of carbon monoxide and hydroxyl radical [cm^3 / s]"""

    kx: pint.Quantity[float] = field(validator=check_units(f"1 / {TIME_UNIT}"))
    """Rate constant for reaction of hydroxyl radical and everything else [1 / s]"""

    tau_dep_h2: pint.Quantity[float] = field(validator=check_units(TIME_UNIT))
    """Partial lifetime of hydrogen due to biogenic soil sinks [s]"""

    air_number: pint.Quantity[float] = field(
        validator=check_units(f"1 / {LENGTH_UNIT}^3"),
        default=UNIT_REGISTRY.Quantity(2.5e19, "1 / cm^3"),
    )
    """Density of air [1 / cm^3]"""

    def solve(
        self,
        emissions: scmdata.ScmRun,
        out_steps: pint.Quantity[np.array],
        y0: Dict[str, pint.Quantity[float]],
        method="Radau",
        max_step=365 * 24 * 60 * 60,
    ):
        scenario = emissions.get_unique_meta("scenario", True)
        model = emissions.get_unique_meta("model", True)

        emms_func = get_emms_func(emissions)
        dconc_dt, jac = self._get_dconc_dt_jac(emms_func)
        out_steps_mag = out_steps.to(TIME_UNIT).magnitude

        y0_h = [y0[c].to(CONC_UNIT).magnitude for c in _CONC_ORDER]

        sol = scipy.integrate.solve_ivp(
            dconc_dt,
            [out_steps_mag[0], out_steps_mag[-1]],
            y0_h,
            method=method,
            vectorized=True,
            max_step=max_step,
            jac=jac,
            t_eval=out_steps_mag,
        )
        #         sol.message

        out_years = pint.Quantity(sol.t, TIME_UNIT).to("year").magnitude

        out = scmdata.ScmRun(
            pd.DataFrame(
                sol.y,
                index=pd.MultiIndex.from_arrays(
                    [
                        [
                            f"Atmospheric Concentrations|{c.upper()}"
                            for c in _CONC_ORDER
                        ],
                        [CONC_UNIT, CONC_UNIT, CONC_UNIT, CONC_UNIT],
                        [
                            "World",
                            "World",
                            "World",
                            "World",
                        ],
                        [model, model, model, model],
                        [
                            scenario,
                            scenario,
                            scenario,
                            scenario,
                        ],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                ),
                columns=out_years,
            )
        )

        return out

    def _get_dconc_dt_jac(self, emms_func):
        per_cm3_to_ppb = self.air_number / UNIT_REGISTRY.Quantity(1e9, "ppb")
        
        k1_mag = (self.k1 * per_cm3_to_ppb).to(RATE_CONSTANT_UNIT).magnitude
        k2_mag = (self.k2 * per_cm3_to_ppb).to(RATE_CONSTANT_UNIT).magnitude
        tau_dep_mag = self.tau_dep_h2.to(TIME_UNIT).magnitude
        k3_mag = (self.k3 * per_cm3_to_ppb).to(RATE_CONSTANT_UNIT).magnitude
        kx_mag = self.kx.to(f"1 / {TIME_UNIT}").magnitude

        def dconc_dt(t, y):
            concs = {k: y[i] for k, i in _CONC_INDEXES.items()}

            emms_ch4_mag, emms_h2_mag, emms_co_mag, emms_oh_mag = emms_func(t)

            dch4dt = emms_ch4_mag - k1_mag * concs["oh"] * concs["ch4"]
            dh2dt = (
                emms_h2_mag
                - k2_mag * concs["oh"] * concs["h2"]
                - concs["h2"] / tau_dep_mag
            )
            dcodt = (
                emms_co_mag
                - k3_mag * concs["oh"] * concs["co"]
                + k1_mag * concs["oh"] * concs["ch4"]
            )
            dohdt = (
                emms_oh_mag
                - kx_mag * concs["oh"]
                - k3_mag * concs["oh"] * concs["co"]
                - k2_mag * concs["oh"] * concs["h2"]
                - k1_mag * concs["oh"] * concs["ch4"]
            )

            out = {"ch4": dch4dt, "h2": dh2dt, "co": dcodt, "oh": dohdt}
            out = [out[c] for c in _CONC_ORDER]

            return out

        def jac(t, y):
            concs = {k: y[i] for k, i in _CONC_INDEXES.items()}

            return [
                [-k1_mag * concs["oh"], 0, 0, -k1_mag * concs["ch4"]],
                [0, -k2_mag * concs["oh"] - 1 / tau_dep_mag, 0, -k2_mag * concs["h2"]],
                [
                    k1_mag * concs["oh"],
                    0,
                    -k3_mag * concs["oh"],
                    -k3_mag * concs["co"] + k1_mag * concs["ch4"],
                ],
                [
                    -k1_mag * concs["oh"],
                    -k2_mag * concs["oh"],
                    -k3_mag * concs["oh"],
                    -kx_mag
                    - k3_mag * concs["co"]
                    - k2_mag * concs["h2"]
                    - k1_mag * concs["ch4"],
                ],
            ]

        return dconc_dt, jac
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# a=emm_values[:, np.newaxis]* np.ones(years.shape)[np.newaxis, :]
# a[2,3]=400
# a
```

```{code-cell} ipython3
def do_experiments(k1,k2, k3, kx
) -> scmdata.run.BaseScmRun:
    """
    Run model experiments
    
    Parameters
    --------
        
    k1
        rate of methane decomposition
    
    k2
        rate of hydrogen decomposition
        
    k3
        rate of carbon monoxide decomposition
    
    kx
        sum of hydroxyl sinks
    
    """
    @define
    class HydrogenBox:
        k1: pint.Quantity[float] = field(
            validator=check_units(f"{LENGTH_UNIT}^3 / {TIME_UNIT}")
        )
        """Rate constant for reaction of methane and hydroxyl radical [cm^3 / s]"""

        k2: pint.Quantity[float] = field(
            validator=check_units(f"{LENGTH_UNIT}^3 / {TIME_UNIT}")
        )
        """Rate constant for reaction of hydrogen and hydroxyl radical [cm^3 / s]"""

        k3: pint.Quantity[float] = field(
            validator=check_units(f"{LENGTH_UNIT}^3 / {TIME_UNIT}")
        )
        """Rate constant for reaction of carbon monoxide and hydroxyl radical [cm^3 / s]"""

        kx: pint.Quantity[float] = field(validator=check_units(f"1 / {TIME_UNIT}"))
        """Rate constant for reaction of hydroxyl radical and everything else [1 / s]"""

        tau_dep_h2: pint.Quantity[float] = field(validator=check_units(TIME_UNIT))
        """Partial lifetime of hydrogen due to biogenic soil sinks [s]"""

        air_number: pint.Quantity[float] = field(
            validator=check_units(f"1 / {LENGTH_UNIT}^3"),
            default=UNIT_REGISTRY.Quantity(2.5e19, "1 / cm^3"),
        )
        """Density of air [1 / cm^3]"""

        def solve(
            self,
            emissions: scmdata.ScmRun,
            out_steps: pint.Quantity[np.array],
            y0: Dict[str, pint.Quantity[float]],
            method="Radau",
            max_step=365 * 24 * 60 * 60,
        ):
            scenario = emissions.get_unique_meta("scenario", True)
            model = emissions.get_unique_meta("model", True)

            emms_func = get_emms_func(emissions)
            dconc_dt, jac = self._get_dconc_dt_jac(emms_func)
            out_steps_mag = out_steps.to(TIME_UNIT).magnitude

            y0_h = [y0[c].to(CONC_UNIT).magnitude for c in _CONC_ORDER]

            sol = scipy.integrate.solve_ivp(
                dconc_dt,
                [out_steps_mag[0], out_steps_mag[-1]],
                y0_h,
                method=method,
                vectorized=True,
                max_step=max_step,
                jac=jac,
                t_eval=out_steps_mag,
            )
            #         sol.message

            out_years = pint.Quantity(sol.t, TIME_UNIT).to("year").magnitude

            out = scmdata.ScmRun(
                pd.DataFrame(
                    sol.y,
                    index=pd.MultiIndex.from_arrays(
                        [
                            [
                                f"Atmospheric Concentrations|{c.upper()}"
                                for c in _CONC_ORDER
                            ],
                            [CONC_UNIT, CONC_UNIT, CONC_UNIT, CONC_UNIT],
                            [
                                "World",
                                "World",
                                "World",
                                "World",
                            ],
                            [model, model, model, model],
                            [
                                scenario,
                                scenario,
                                scenario,
                                scenario,
                            ],
                        ],
                        names=["variable", "unit", "region", "model", "scenario"],
                    ),
                    columns=out_years,
                )
            )

            return out

        def _get_dconc_dt_jac(self, emms_func):
            per_cm3_to_ppb = self.air_number / UNIT_REGISTRY.Quantity(1e9, "ppb")

            k1_mag = (self.k1 * per_cm3_to_ppb).to(RATE_CONSTANT_UNIT).magnitude
            k2_mag = (self.k2 * per_cm3_to_ppb).to(RATE_CONSTANT_UNIT).magnitude
            tau_dep_mag = self.tau_dep_h2.to(TIME_UNIT).magnitude
            k3_mag = (self.k3 * per_cm3_to_ppb).to(RATE_CONSTANT_UNIT).magnitude
            kx_mag = self.kx.to(f"1 / {TIME_UNIT}").magnitude

            def dconc_dt(t, y):
                concs = {k: y[i] for k, i in _CONC_INDEXES.items()}

                emms_ch4_mag, emms_h2_mag, emms_co_mag, emms_oh_mag = emms_func(t)

                dch4dt = emms_ch4_mag - k1_mag * concs["oh"] * concs["ch4"]
                dh2dt = (
                    emms_h2_mag
                    - k2_mag * concs["oh"] * concs["h2"]
                    - concs["h2"] / tau_dep_mag
                )
                dcodt = (
                    emms_co_mag
                    - k3_mag * concs["oh"] * concs["co"]
                    + k1_mag * concs["oh"] * concs["ch4"]
                )
                dohdt = (
                    emms_oh_mag
                    - kx_mag * concs["oh"]
                    - k3_mag * concs["oh"] * concs["co"]
                    - k2_mag * concs["oh"] * concs["h2"]
                    - k1_mag * concs["oh"] * concs["ch4"]
                )

                out = {"ch4": dch4dt, "h2": dh2dt, "co": dcodt, "oh": dohdt}
                out = [out[c] for c in _CONC_ORDER]

                return out

            def jac(t, y):
                concs = {k: y[i] for k, i in _CONC_INDEXES.items()}

                return [
                    [-k1_mag * concs["oh"], 0, 0, -k1_mag * concs["ch4"]],
                    [0, -k2_mag * concs["oh"] - 1 / tau_dep_mag, 0, -k2_mag * concs["h2"]],
                    [
                        k1_mag * concs["oh"],
                        0,
                        -k3_mag * concs["oh"],
                        -k3_mag * concs["co"] + k1_mag * concs["ch4"],
                    ],
                    [
                        -k1_mag * concs["oh"],
                        -k2_mag * concs["oh"],
                        -k3_mag * concs["oh"],
                        -kx_mag
                        - k3_mag * concs["co"]
                        - k2_mag * concs["h2"]
                        - k1_mag * concs["ch4"],
                    ],
                ]

            return dconc_dt, jac
        
    # input emissions
    years = np.arange(1000, 1501)

    emm_values = np.array([204.5, 259.7, 223.7, 1464])
    input_emms = scmdata.ScmRun(
        pd.DataFrame(
            emm_values[:, np.newaxis]
            * np.ones(years.shape)[np.newaxis, :],
            index=pd.MultiIndex.from_arrays(
                [
                    ["Emissions|CH4", "Emissions|CO", "Emissions|H2", "Emissions|OH"],
                    ["ppb / yr", "ppb / yr", "ppb / yr", "ppb / yr"],
                    [
                        "World",
                        "World",
                        "World",
                        "World",
                    ],
                    [
                        "unspecified",
                        "unspecified",
                        "unspecified",
                        "unspecified",
                    ],
                    [
                        "Prather eqm",
                        "Prather eqm",
                        "Prather eqm",
                        "Prather eqm",
                    ],
                ],
                names=["variable", "unit", "region", "model", "scenario"],
            ),
            columns=years,
        )
    )

    if years.size <100:
        shift_emms = input_emms.copy()

    else:
        temp = emm_values[:, np.newaxis]* np.ones(years.shape)[np.newaxis, :]
        temp2 = temp.copy()
        temp3 = temp.copy()
        temp4 = temp.copy()
        
        n_years=72
        for year in range(n_years):
            temp[2, 50 + year + 1]=1.01*temp[2,50 + year]
            
            temp2[2, 50 + year + 1]=1.005*temp2[2,50 + year]
            
            temp3[2, 50 + year + 1]=1.01*temp3[2,50 + year]
            
            temp4[0, 50 + year + 1]=1.01*temp4[0,50 + year]
            temp4[1, 50 +  year + n_years + 1]=1.01*temp4[1,50 + year+ n_years]
            temp4[2, 50+ year + 2 * n_years + 1]=1.01*temp4[2,50+ year+ 2* n_years]

        for year in range(n_years*2):
            temp3[0,50 + year + 1]=temp3[0,50 + year]/1.01

        temp[2,50+n_years+1:]=temp[2,50+n_years]
        
        temp2[2,50+n_years+1:]=temp2[2,50+n_years]
        
        temp3[2,50+n_years+1:]=temp3[2,50+n_years]
        temp3[0,50+n_years*2+1:]=temp3[0,50+n_years*2]

        temp4[0,50+n_years+1:]=temp4[0,50+n_years]
        temp4[1,50+n_years*2+1:]=temp4[1,50+n_years*2]
        temp4[1,50+n_years*3+1:]=temp4[1,50+n_years*3]

        shift_emms = scmdata.ScmRun(
            pd.DataFrame(
                temp,
                index=pd.MultiIndex.from_arrays(
                    [
                        ["Emissions|CH4", "Emissions|CO", "Emissions|H2", "Emissions|OH"],
                        ["ppb / yr", "ppb / yr", "ppb / yr", "ppb / yr"],
                        [
                            "World",
                            "World",
                            "World",
                            "World",
                        ],
                        [
                            "unspecified",
                            "unspecified",
                            "unspecified",
                            "unspecified",
                        ],
                        [
                            "Increase H2",
                            "Increase H2",
                            "Increase H2",
                            "Increase H2",
                        ],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                ),
                columns=years,
            )
        )
#         input_emms = scmdata.ScmRun(
#             pd.DataFrame(
#                 temp2,
#                 index=pd.MultiIndex.from_arrays(
#                     [
#                         ["Emissions|CH4", "Emissions|CO", "Emissions|H2", "Emissions|OH"],
#                         ["ppb / yr", "ppb / yr", "ppb / yr", "ppb / yr"],
#                         [
#                             "World",
#                             "World",
#                             "World",
#                             "World",
#                         ],
#                         [
#                             "unspecified",
#                             "unspecified",
#                             "unspecified",
#                             "unspecified",
#                         ],
#                         [
#                             "Small H2",
#                             "Small H2",
#                             "Small H2",
#                             "Small H2",
#                         ],
#                     ],
#                     names=["variable", "unit", "region", "model", "scenario"],
#                 ),
#                 columns=years,
#             )
#         )
    input_emms = scmdata.ScmRun(
            pd.DataFrame(
                temp4,
                index=pd.MultiIndex.from_arrays(
                    [
                        ["Emissions|CH4", "Emissions|CO", "Emissions|H2", "Emissions|OH"],
                        ["ppb / yr", "ppb / yr", "ppb / yr", "ppb / yr"],
                        [
                            "World",
                            "World",
                            "World",
                            "World",
                        ],
                        [
                            "unspecified",
                            "unspecified",
                            "unspecified",
                            "unspecified",
                        ],
                        [
                            "Increase H2 Lower CH4",
                            "Increase H2 Lower CH4",
                            "Increase H2 Lower CH4",
                            "Increase H2 Lower CH4",
                        ],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                ),
                columns=years,
            )
        )


    y0 = {
        "ch4": UNIT_REGISTRY.Quantity(1851.71, "ppb"),
        "co": UNIT_REGISTRY.Quantity(132.4, "ppb"),
        "h2": UNIT_REGISTRY.Quantity(432.33, "ppb"),
        "oh": UNIT_REGISTRY.Quantity(2.22e-05, "ppb"),
    } 
    
    to_solve = HydrogenBox(
    k1=k1,
    k2=k2,
    k3=k3,
    kx=kx,
    tau_dep_h2=UNIT_REGISTRY.Quantity(2.5, "year"),
    )
    scens_res=[]
    
    time_axis_m=years
    out_steps = UNIT_REGISTRY.Quantity(input_emms.time_points.years(), "year")

    for name, to_solve_l, y0, emms_l in (
        (
            "constant_emms",
            to_solve,
            y0,
            input_emms,
        ),
        ("shift_emms", to_solve, y0, shift_emms),
    ):
        res = to_solve_l.solve(emms_l,out_steps=out_steps,y0=y0)
        scens_res.append(res)
#         if not res[name].success:
#             raise ValueError("Model failed to solve")

#     out = scmdata.run.BaseScmRun(
#         pd.DataFrame(
#             np.vstack([res["constant_emms"].y[0, :], res["shift_emms"].y[0, :]]),
#             index=pd.MultiIndex.from_arrays(
#                 (
#                     ["position", "position"],
#                     [LENGTH_UNITS, LENGTH_UNITS],
#                     ["constant_emms", "shift_emms"],
#                 ),
#                 names=["variable", "unit", "scenario"],
#             ),
#             columns=time_axis.to(TIME_UNITS).m,
#         )
#     )

#     out["model"] = "example"
    out = scmdata.run_append(scens_res)
    return out
```

### Target

Use Jpl values to calculate a target.

```{code-cell} ipython3
k1_jpl = UNIT_REGISTRY.Quantity(6.3e-15, "cm^3 / s")
k2_jpl = UNIT_REGISTRY.Quantity(6.7e-15, "cm^3 / s")  # JPL publication 19-5, page 1-53
k3_jpl = UNIT_REGISTRY.Quantity(2e-13, "cm^3 / s")
kx_base = UNIT_REGISTRY.Quantity(1.062, "1 / s")
y0 = {
    "ch4": UNIT_REGISTRY.Quantity(1785, "ppb"),
    "co": UNIT_REGISTRY.Quantity(98, "ppb"),
    "h2": UNIT_REGISTRY.Quantity(500, "ppb"),
    "oh": UNIT_REGISTRY.Quantity(2e-5, "ppb"),
}
```

```{code-cell} ipython3
truth = {
    "k1" : UNIT_REGISTRY.Quantity(6.3e-15, "cm^3 / s"),
    "k2" : UNIT_REGISTRY.Quantity(6.7e-15, "cm^3 / s") ,
    "k3" : UNIT_REGISTRY.Quantity(2e-13, "cm^3 / s"),
    "kx" :UNIT_REGISTRY.Quantity(1.062, "1 / s"),
}
# truth = {
#     "k1" : UNIT_REGISTRY.Quantity(6.3e-2, "cm^3 / s"),
#     "k2" : UNIT_REGISTRY.Quantity(6.7e-2, "cm^3 / s") ,
#     "k3" : UNIT_REGISTRY.Quantity(2e-2, "cm^3 / s"),
#     "kx" :UNIT_REGISTRY.Quantity(1.062, "1 / s"),
# }

target = do_experiments(**truth)
target["model"] = "target"
# target.lineplot(time_axis="year-month")

for vdf in target.groupby("variable"):
    vdf.lineplot(style="variable")
    plt.show()
```

```{code-cell} ipython3
target
```

### Cost calculation

The next thing is to decide how we're going to calculate the cost function. There are many options here, in this case we're going to use the sum of squared errors.

```{code-cell} ipython3

normalisation = pd.Series(
#     [0.1,0.1,0.1,0.1],
    [1,1,1,1],

    index=pd.MultiIndex.from_arrays(
        (
            [
                "Atmospheric Concentrations|CH4",
                "Atmospheric Concentrations|H2",
                "Atmospheric Concentrations|CO",
                "Atmospheric Concentrations|OH",
            ],
            ["ppb","ppb","ppb","ppb"],
        ),
        names=["variable", "unit"],
    ),
)
normalisation
```

```{code-cell} ipython3

cost_calculator = OptCostCalculatorSSE.from_series_normalisation(
    target=target, normalisation_series=normalisation, model_col="model"
)
assert cost_calculator.calculate_cost(target) == 0
assert cost_calculator.calculate_cost(target * 1.1) > 0
cost_calculator
```

### Model runner

Scipy does everything using numpy arrays. Here we use a wrapper that converts them to pint quantities before running.

+++

Firstly, we define the parameters we're going to optimise. This will be used to ensure a consistent order throughout.

```{code-cell} ipython3
parameters = [
    ("k1", f"{LENGTH_UNIT}^3 / {TIME_UNIT}"),
    ("k2", f"{LENGTH_UNIT}^3 / {TIME_UNIT}"),
    ("k3", f"{LENGTH_UNIT}^3 / {TIME_UNIT}"),
    ("kx",f"1 / {TIME_UNIT}")
]
parameters
```

Next we define a function which, given pint quantities, returns the inputs needed for our `do_experiments` function. In this case this is not a very interesting function, but in other use cases the flexibility is helpful.

```{code-cell} ipython3
def do_model_runs_input_generator(
    k1: pint.Quantity, k2: pint.Quantity, k3: pint.Quantity, kx: pint.Quantity
) -> Dict[str, pint.Quantity]:
    """
    Create the inputs for :func:`do_experiments`

    Parameters
    ----------
    k1
        k1

    k2
        k2

    k3
        k3
        
    kx
        kx

    Returns
    -------
        Inputs for :func:`do_experiments`
    """
    return {"k1": k1, "k2": k2, "k3": k3, "kx": kx}
```

```{code-cell} ipython3
model_runner = OptModelRunner.from_parameters(
    params=parameters,
    do_model_runs_input_generator=do_model_runs_input_generator,
    do_model_runs=do_experiments,
)
model_runner
```

Now we can run from a plain numpy array (like scipy will use) and get a result that will be understood by our cost calculator.

```{code-cell} ipython3
cost_calculator.calculate_cost(model_runner.run_model([1e-15, 1e-15, 1e-13,1]))
```

Now we're ready to optimise.

+++

## Global optimisation

Scipy has many [global optimisation options](https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization). Here we show how to do this with differential evolution, but using others would be equally simple.

+++

We have to define where to start the optimisation.

```{code-cell} ipython3
start = np.array([1e-15, 1e-15, 1e-13,1])
start
```

For this optimisation, we must also define bounds for each parameter.
```
truth = {
    "k1" : UNIT_REGISTRY.Quantity(6.3e-15, "cm^3 / s"),
    "k2" : UNIT_REGISTRY.Quantity(6.7e-15, "cm^3 / s") ,
    "k3" : UNIT_REGISTRY.Quantity(2e-13, "cm^3 / s"),
    "kx" :UNIT_REGISTRY.Quantity(1.062, "1 / s"),
}
```

```{code-cell} ipython3
bounds_dict = {
    "k1": [
        UREG.Quantity(1e-15, "cm^3 / s"),
        UREG.Quantity(1e-14, "cm^3 / s"),
    ],
    "k2": [
        UREG.Quantity(1e-15, "cm^3 / s"),
        UREG.Quantity(1e-14, "cm^3 / s"),
    ],
    "k3": [
        UREG.Quantity(1e-13, "cm^3 / s"),
        UREG.Quantity(1e-12, "cm^3 / s"),
    ],
    
    "kx": [
        UREG.Quantity(0.4, "1 / s"),
        UREG.Quantity(2, "1 / s"),
    ],
 
}
display(bounds_dict)

bounds = [[v.to(unit).m for v in bounds_dict[k]] for k, unit in parameters]
bounds
```

Now we're ready to run our optimisation.

```{code-cell} ipython3
# Number of parallel processes to use
processes = 4

# Random seed (use if you want reproducibility)
seed = 12849

## Optimisation parameters - here we use short runs
## TODO: other repo with full runs
# Tolerance to set for convergance
atol = 1
tol = 0.02
# Maximum number of iterations to use
maxiter = 8
# Lower mutation means faster convergence but smaller
# search radius
mutation = (0.1, 0.8)
# Higher recombination means faster convergence but
# might miss global minimum
recombination = 0.8
# Size of population to use (higher number means more searching
# but slower convergence)
popsize = 4
# There are also the strategy and init options
# which might be needed for some problems

# Maximum number of runs to store
max_n_runs = (maxiter + 1) * popsize * len(parameters)


# Visualisation options
update_every = 4
thin_ts_to_plot = 5


# Create axes to plot on (could also be created as part of a factory
# or class method)
convert_scmrun_to_plot_dict = partial(scmrun_as_dict, groups=["variable", "scenario"])

cost_name = "cost"
timeseries_axes = list(convert_scmrun_to_plot_dict(target).keys())

parameters_names = [v[0] for v in parameters]
parameters_mosaic = list(more_itertools.repeat_each(parameters_names, 1))
timeseries_axes_mosaic = list(more_itertools.repeat_each(timeseries_axes, 1))

fig, axd = plt.subplot_mosaic(
    mosaic=[
        [cost_name] + timeseries_axes_mosaic[0:4],
        [cost_name]+ timeseries_axes_mosaic[4:8],
        [cost_name]+parameters_mosaic,
    ],
    figsize=(12, 6),
)
holder = display(fig, display_id=True)

with Manager() as manager:
    store = OptResStore.from_n_runs_manager(
        max_n_runs,
        manager,
        params=parameters_names,
    )


    # Create objects and functions to use
    to_minimize = partial(
        to_minimize_full,
        store=store,
        cost_calculator=cost_calculator,
        model_runner=model_runner,
        known_error=ValueError,
    )

    with manager.Pool(processes=processes) as pool:
        with tqdm(total=max_n_runs) as pbar:
            opt_plotter = OptPlotter(
                holder=holder,
                fig=fig,
                axes=axd,
                cost_key=cost_name,
                parameters=parameters_names,
                timeseries_axes=timeseries_axes,
                convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
                target=target,
                store=store,
                thin_ts_to_plot=thin_ts_to_plot,
            )

            proxy = CallbackProxy(
                real_callback=opt_plotter,
                store=store,
                update_every=update_every,
                progress_bar=pbar,
                last_callback_val=0,
            )

            # This could be wrapped up too
            optimize_res = scipy.optimize.differential_evolution(
                to_minimize,
                bounds,
                maxiter=maxiter,
                x0=start,
                tol=tol,
                atol=atol,
                seed=seed,
                # Polish as a second step if you want
                polish=False,
                workers=pool.map,
                updating="deferred",  # as we run in parallel, this has to be used
                mutation=mutation,
                recombination=recombination,
                popsize=popsize,
                callback=proxy.callback_differential_evolution,
            )

plt.close()
optimize_res
```

```{code-cell} ipython3
[cost_name] + timeseries_axes_mosaic[0:4]
```

```{code-cell} ipython3
fig, axd = plt.subplot_mosaic(
    mosaic=[
        [cost_name] + timeseries_axes_mosaic[0:4],
        [cost_name]+ timeseries_axes_mosaic[4:8],
        [cost_name]+parameters_mosaic,
    ],
    figsize=(12, 6),
)
holder = display(fig, display_id=True)

with Manager() as manager:
    store = OptResStore.from_n_runs_manager(
        max_n_runs,
        manager,
        params=parameters_names,
    )


    # Create objects and functions to use
    to_minimize = partial(
        to_minimize_full,
        store=store,
        cost_calculator=cost_calculator,
        model_runner=model_runner,
        known_error=ValueError,
    )

    with manager.Pool(processes=processes) as pool:
        with tqdm(total=max_n_runs) as pbar:
            opt_plotter = OptPlotter(
                holder=holder,
                fig=fig,
                axes=axd,
                cost_key=cost_name,
                parameters=parameters_names,
                timeseries_axes=timeseries_axes,
                convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
                target=target,
                store=store,
                thin_ts_to_plot=thin_ts_to_plot,
            )

            proxy = CallbackProxy(
                real_callback=opt_plotter,
                store=store,
                update_every=update_every,
                progress_bar=pbar,
                last_callback_val=0,
            )

            # This could be wrapped up too
            optimize_res = scipy.optimize.differential_evolution(
                to_minimize,
                bounds,
                maxiter=maxiter,
                x0=start,
                tol=tol,
                atol=atol,
                seed=seed,
                # Polish as a second step if you want
                polish=False,
                workers=pool.map,
                updating="deferred",  # as we run in parallel, this has to be used
                mutation=mutation,
                recombination=recombination,
                popsize=popsize,
                callback=proxy.callback_differential_evolution,
            )

plt.close()
optimize_res
```

```
truth = {
    "k1" : UNIT_REGISTRY.Quantity(6.3e-15, "cm^3 / s"),
    "k2" : UNIT_REGISTRY.Quantity(6.7e-15, "cm^3 / s") ,
    "k3" : UNIT_REGISTRY.Quantity(2e-13, "cm^3 / s"),
    "kx" :UNIT_REGISTRY.Quantity(1.062, "1 / s"),
}
```

+++

## Local optimisation

Scipy also has [local optimisation](https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization) (e.g. Nelder-Mead) options. Here we show how to do this.

+++

Again, we have to define where to start the optimisation (this has a greater effect on local optimisation).

```{code-cell} ipython3
with Manager() as manager:
    store = OptResStore.from_n_runs_manager(
        max_n_runs,
        manager,
        params=parameters_names,
    )


    # Create objects and functions to use
    to_minimize = partial(
        to_minimize_full,
        store=store,
        cost_calculator=cost_calculator,
        model_runner=model_runner,
        known_error=ValueError,
    )

    with manager.Pool(processes=processes) as pool:
        with tqdm(total=max_n_runs) as pbar:
            opt_plotter = OptPlotter(
                holder=holder,
                fig=fig,
                axes=axd,
                cost_key=cost_name,
                parameters=parameters_names,
                timeseries_axes=timeseries_axes,
                convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
                target=target,
                store=store,
                thin_ts_to_plot=thin_ts_to_plot,
            )

            proxy = CallbackProxy(
                real_callback=opt_plotter,
                store=store,
                update_every=update_every,
                progress_bar=pbar,
                last_callback_val=0,
            )

            # This could be wrapped up too
            optimize_res = scipy.optimize.differential_evolution(
                to_minimize,
                bounds,
                maxiter=maxiter,
                x0=start,
                tol=tol,
                atol=atol,
                seed=seed,
                # Polish as a second step if you want
                polish=False,
                workers=pool.map,
                updating="deferred",  # as we run in parallel, this has to be used
                mutation=mutation,
                recombination=recombination,
                popsize=popsize,
                callback=proxy.callback_differential_evolution,
            )

plt.close()
optimize_res
```

```{code-cell} ipython3
# Here we imagine that we're polishing from the results of the DE above,
# but we make the start slightly worse first
start_local = optimize_res.x
start_local
```

```{code-cell} ipython3
# Optimisation parameters
tol = 1e-4
# Maximum number of iterations to use
maxiter = 30

# I think this is how this works
max_n_runs = len(parameters) + 2 * maxiter

# Lots of options here
method = "Nelder-mead"

# Visualisation options
update_every = 10
thin_ts_to_plot = 5
parameters_names = [v[0] for v in parameters]

# Create other objects
store = OptResStore.from_n_runs(max_n_runs, params=parameters_names)
to_minimize = partial(
    to_minimize_full,
    store=store,
    cost_calculator=cost_calculator,
    model_runner=model_runner,
)


with tqdm(total=max_n_runs) as pbar:
    # Here we use a class method which auto-generates the figure
    # for us. This is just a convenience thing, it does the same
    # thing as the previous example under the hood.
    opt_plotter = OptPlotter.from_autogenerated_figure(
        cost_key=cost_name,
        params=parameters_names,
        convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
        target=target,
        store=store,
        thin_ts_to_plot=thin_ts_to_plot,
        kwargs_create_mosaic=dict(
            n_parameters_per_row=3,
            n_timeseries_per_row=1,
            cost_col_relwidth=2,
        ),
        kwargs_get_fig_axes_holder=dict(figsize=(10, 12)),
        plot_cost_kwargs={
            "alpha": 0.7,
            "get_ymax": partial(
                get_ymax_default, min_scale_factor=1e6, min_v_median_scale_factor=0
            ),
        },
    )

    proxy = CallbackProxy(
        real_callback=opt_plotter,
        store=store,
        update_every=update_every,
        progress_bar=pbar,
        last_callback_val=0,
    )

    optimize_res_local = scipy.optimize.minimize(
        to_minimize,
        x0=start_local,
        tol=tol,
        method=method,
        options={"maxiter": maxiter},
        callback=proxy.callback_minimize,
    )

plt.close()
optimize_res_local
```

## MCMC

To run MCMC, we use the [emcee](https://emcee.readthedocs.io/) package. This has heaps of options for running MCMC and is really user friendly. All the different available moves/samplers are listed [here](https://emcee.readthedocs.io/en/stable/user/moves/).

```{code-cell} ipython3
def neg_log_prior_bounds(x: np.ndarray, bounds: np.ndarray) -> float:
    """
    Log prior that just checks proposal is in bounds

    Parameters
    ----------
    x
        Parameter array

    bounds
        Bounds for each parameter (must have same
        order as x)
    """
    in_bounds = (x > bounds[:, 0]) & (x < bounds[:, 1])
    if np.all(in_bounds):
        return 0

    return -np.inf


neg_log_prior = partial(neg_log_prior_bounds, bounds=np.array(bounds))
```

```{code-cell} ipython3
def log_prob(x) -> Tuple[float, float, float]:
    neg_ll_prior_x = neg_log_prior(x)

    if not np.isfinite(neg_ll_prior_x):
        return -np.inf, None, None

    try:
        model_results = model_runner.run_model(x)
    except ValueError:
        return -np.inf, None, None

    sses = cost_calculator.calculate_cost(model_results)
    neg_ll_x = -sses / 2
    ll = neg_ll_x + neg_ll_prior_x

    return ll, neg_ll_prior_x, neg_ll_x
```

We're using the DIME proposal from [emcwrap](https://github.com/gboehl/emcwrap). This claims to have an adaptive proposal distribution so requires less fine tuning and is less sensitive to the starting point.

```{code-cell} ipython3
ndim = len(bounds)
# emcwrap docs suggest 5 * ndim
nwalkers = 5 * ndim

start_emcee = [s + s / 100 * np.random.rand(nwalkers) for s in optimize_res_local.x]
start_emcee = np.vstack(start_emcee).T

move = DIMEMove()
```

```{code-cell} ipython3
# Use HDF5 backend
filename = "basic-demo-mcmc.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
```

```{code-cell} ipython3
# How many parallel process to use
processes = 4

# Set the seed to ensure reproducibility
np.random.seed(424242)

## MCMC options
# Unclear at the start how many iterations are needed to sample
# the posterior appropriately, normally requires looking at the
# chains and then just running them for longer if needed.
# This number is definitely too small
max_iterations = 200
burnin = 10
thin = 2

## Visualisation options
plot_every = 15
convergence_ratio = 50
parameter_order = [p[0] for p in parameters]
neg_log_likelihood_name = "neg_ll"
labels_chain = [neg_log_likelihood_name] + parameter_order

# Stores for autocorr over steps
autocorr = np.zeros(max_iterations)
autocorr_steps = np.zeros(max_iterations)
index = 0

## Setup plots
fig_chain, axd_chain = plt.subplot_mosaic(
    mosaic=[[l] for l in labels_chain],
    figsize=(10, 5),
)
holder_chain = display(fig_chain, display_id=True)

fig_dist, axd_dist = plt.subplot_mosaic(
    mosaic=[[l] for l in parameter_order],
    figsize=(10, 5),
)
holder_dist = display(fig_dist, display_id=True)

fig_corner = plt.figure(figsize=(6, 6))
holder_corner = display(fig_dist, display_id=True)

fig_tau, ax_tau = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(4, 4),
)
holder_tau = display(fig_tau, display_id=True)

# Plottting helper
truths_corner = [truth[k].to(u).m for k, u in parameters]

with Pool(processes=processes) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob,
        moves=move,
        backend=backend,
        blobs_dtype=[("neg_log_prior", float), ("neg_log_likelihood", float)],
        pool=pool,
    )

    for sample in sampler.sample(
        # If statement in case we're continuing a run rather than starting fresh
        start_emcee if sampler.iteration < 1 else sampler.get_last_sample(),
        iterations=max_iterations,
        progress="notebook",
        progress_kwargs={"leave": True},
    ):
        if sampler.iteration % plot_every or sampler.iteration < 2:
            continue

        if sampler.iteration < burnin + 1:
            in_burn_in = True
        else:
            in_burn_in = False

        for ax in axd_chain.values():
            ax.clear()

        emcee_plotting.plot_chains(
            inp=sampler,
            burnin=burnin,
            parameter_order=parameter_order,
            axes_d=axd_chain,
            neg_log_likelihood_name=neg_log_likelihood_name,
        )
        fig_chain.tight_layout()
        holder_chain.update(fig_chain)

        if not in_burn_in:
            chain_post_burnin = sampler.get_chain(discard=burnin)
            if chain_post_burnin.shape[0] > 0:
                acceptance_fraction = np.mean(
                    get_acceptance_fractions(chain_post_burnin)
                )
                print(
                    f"{chain_post_burnin.shape[0]} steps post burnin, "
                    f"acceptance fraction: {acceptance_fraction}"
                )

            for ax in axd_dist.values():
                ax.clear()

            emcee_plotting.plot_dist(
                inp=sampler,
                burnin=burnin,
                thin=thin,
                parameter_order=parameter_order,
                axes_d=axd_dist,
                warn_singular=False,
            )
            fig_dist.tight_layout()
            holder_dist.update(fig_dist)

            try:
                fig_corner.clear()
                emcee_plotting.plot_corner(
                    inp=sampler,
                    burnin=burnin,
                    thin=thin,
                    parameter_order=parameter_order,
                    fig=fig_corner,
                    truths=truths_corner,
                )
                fig_corner.tight_layout()
                holder_corner.update(fig_corner)
            except AssertionError:
                pass

            autocorr_bits = get_autocorrelation_info(
                sampler,
                burnin=burnin,
                thin=thin,
                autocorr_tol=0,
                convergence_ratio=convergence_ratio,
            )
            autocorr[index] = autocorr_bits["autocorr"]
            autocorr_steps[index] = sampler.iteration - burnin
            index += 1

            if np.sum(autocorr > 0) > 1 and np.sum(~np.isnan(autocorr)) > 1:
                # plot autocorrelation, pretty specific to setup so haven't
                # created separate function
                ax_tau.clear()
                ax_tau.plot(
                    autocorr_steps[:index],
                    autocorr[:index],
                )
                ax_tau.axline(
                    (0, 0), slope=1 / convergence_ratio, color="k", linestyle="--"
                )
                ax_tau.set_ylabel("Mean tau")
                ax_tau.set_xlabel("Number steps (post burnin)")
                holder_tau.update(fig_tau)

# Close all the figures
for _ in range(4):
    plt.close()
```
