---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
# %matplotlib inline
```

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
```

```{code-cell} ipython3
import pandas as pd
import pint
import scipy.integrate
import scmdata.run
from emcwrap import DIMEMove
from multiprocess import Pool, Manager
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

# Import data

```{code-cell} ipython3
# fname="datasets/historical_gas_conc.nc"
# loaded_nc = scmdata.ScmRun.from_nc(fname=fname)
# loaded_nc
# for vdf in loaded_nc.groupby("variable"):
#     vdf.lineplot()
#     plt.show()
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

We're going to calibrate the model's response to historical data experiments:

+++

# Import concentrations and emissions

+++

### 20th Century Concentrations

+++

## CO

```{code-cell} ipython3
years_co= np.arange(1992,2023)



def load_noaa_csv(paths,gases,headers):
    df =[]
    for path, gas, header in zip(paths,gases,headers):
        daily= pd.read_csv(path, header=header,delim_whitespace=True)
        
        #Filter data
        daily.loc[daily["qcflag"]!='...',"value"]=None
        daily=daily.interpolate(method='linear')
        yearly = daily.groupby("year")["value"].mean()
        df.append(yearly)
    return df


gases = [ "CO"]

noaa_paths_nh =[
                'datasets/co_mhd_surface-flask_1_ccgg_event.txt',
               ]
headers = [152]

noaa_nh = load_noaa_csv(noaa_paths_nh,gases,headers)

noaa_paths_sh = ['datasets/co_cgo_surface-flask_1_ccgg_event.txt']
headers = [ 160]

noaa_sh = load_noaa_csv(noaa_paths_sh,gases,headers)
noaa_global = (noaa_nh[0]+noaa_sh[0])/2
noaa_global=noaa_global.loc[years_co]

noaa_global
```

```{code-cell} ipython3
dt.datetime(2013,1,1)
```

```{code-cell} ipython3
noaa_global.index.values
```

```{code-cell} ipython3

co_concentrations= scmdata.ScmRun(
    pd.DataFrame(
      np.nan * np.ones(years.shape)[np.newaxis, :],
        index=pd.MultiIndex.from_arrays(
            [
                [ "Atmospheric Concentrations|CO"],
                ["ppb"],
                [
                    "World",
                ],
                [
                    "None",
                ],
                [
                    "historical",
                ],
            ],
            names=["variable", "unit", "region", "model", "scenario"],
        ),
        columns=years,
    )
)

# def add_co_concentrations(scen,concentrations,years_co):
def add_co_concentrations(scen,concentrations,start=1992,end=2023):
    out=scen.copy().timeseries()
    out.loc[out.index.get_level_values("variable").isin(["Atmospheric Concentrations|CO"]),dt.datetime(start,1,1):dt.datetime(end,1,1)]=noaa_global.values
    return scmdata.ScmRun(out)

co_concentrations=add_co_concentrations(co_concentrations,noaa_global,start=1992,end=2023)
for vdf in co_concentrations.groupby("variable"):
    vdf.lineplot(style="variable")
    plt.show()
    

# concentrations=concentrations.append(oh_concentrations)
# co_concentrations.timeseries()
```

### CH4

```{code-cell} ipython3
years = np.arange(1940,2023)
def get_ch4_conc(path,years):
    out = pd.read_csv(path,header=0)
    out = out.loc[out["year"].isin(years),["year","data_mean_global"]]
    
    return out


path_old = 'datasets/methane.csv/historical/CMIP6GHGConcentrationHistorical_1_2_0/mole-fraction-of-methane-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.csv'
path_new= 'datasets/methane.csv/future/CMIP6GHGConcentrationProjections_1_2_1/mole-fraction-of-methane-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.csv'


ch4_conc = pd.concat([get_ch4_conc(path_old,years),get_ch4_conc(path_new,years)])
ch4_conc = ch4_conc.set_index('year')
ch4_conc
```

### Hydrogen

```{code-cell} ipython3

```

```{code-cell} ipython3
def load_gas_csv(paths,gases,header):
    df =[]
    for path, gas in zip(paths,gases):
        daily= pd.read_csv(path, header=header,delim_whitespace=True)
        daily.mole_fraction.replace({np.nan: None}, inplace=True)
        daily.loc[daily["flag"]!="B","mole_fraction"]=None
        daily=daily.interpolate(method='linear')
        yearly = daily.groupby("YYYY")["mole_fraction"].mean()
        df.append(yearly)
    return df


# import agage
agage_path_nh = ['datasets/AGAGE-GCMD_MHD_h2.txt']

agage_path_sh = ['datasets/AGAGE-GCMD_CGO_h2.txt']
gases = ["H2"]

agage_nh = load_gas_csv(agage_path_nh,gases,header=17)
agage_sh = load_gas_csv(agage_path_sh,gases,header=16)

agage_global = [(x+y)/2 for x,y in zip(agage_nh,agage_sh)]
```

### Create historical

Concentrations increased by 1.5 ppb/yr betwen 1910 and 1952, then increased by 2.7 ppb/yr
```
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL087787
```

```{code-cell} ipython3
# def add_trend(conc, start_year,rate):
#     years= np.arange(start_year,conc.index[0])
#     print(years)
#     print()
#     out = [conc.values[0]-rate*(start_year-year) for year in conc.index] 
#     return out
# conc= pd.DataFrame(np.arange(10,15)[:,np.newaxis])
# add_trend(conc,-3,10)

def create_trend(rate, end_value, start,end):
    years=np.arange(start,end)
    trend = rate*(years-end) + end_value
    return trend,years

end_year= 1994
middle_year = 1952
start_year=years[0]
trend2, years2= create_trend(2.7,agage_global[0].loc[1994],middle_year,end_year)
trend1,years1 = create_trend(1.5,trend2[0],start_year,middle_year)
h2_values = np.concatenate((trend1,trend2,agage_global[0].loc[end_year:]))
```

### Combine concentrations

```{code-cell} ipython3
combined_concentrations=[ch4_conc["data_mean_global"].values,h2_values]
```

```{code-cell} ipython3
concentrations = scmdata.ScmRun(
         pd.DataFrame(
                combined_concentrations,
                index=pd.MultiIndex.from_arrays(
                    [
                        [
                           "Atmospheric Concentrations|CH4",
                            "Atmospheric Concentrations|H2",
                        ],
                        [
                            "ppb",
                            "ppb"
                        ],
                        [
                           
                            "World",
                            "World",
                        ],
                        [
                            "None",
                            "None",
                        ],
                        [
                            "historical",
                            "historical",
                        
                        ],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                ),
                columns=years,
            )
)
concentrations=concentrations.append(co_concentrations)
# for vdf in concentrations.groupby("variable"):
#     vdf.lineplot(hue="variable")
concentrations.lineplot(hue='variable')
```

```{code-cell} ipython3
# combined_concentrations
```

## Emissions

```{code-cell} ipython3
atmosphere_molar_mass = UNIT_REGISTRY.Quantity(28.97, "g / mol")# UNIT_REGISTRY.Quantity(28.97, "g / mol")  # https://www.cs.mcgill.ca/~rwest/wikispeedia/wpcd/wp/e/Earth%2527s_atmosphere.htm#:~:text=The%20mean%20molar%20mass%20of%20air%20is%2028.97%20g%2Fmol.
atmosphere_mass =UNIT_REGISTRY.Quantity(4.22e18,'kg' )# This value is the tropospheric value. 
# UNIT_REGISTRY.Quantity(5.18e18, "kg") 
atmosphere_mole_outer = atmosphere_mass / atmosphere_molar_mass
atmosphere_mole_outer

# def to_ppb_yr(
#     emms,
#     molar_mass,
#     atmosphere_mole=atmosphere_mole_outer,
# ):
#     emms_mole = emms / molar_mass
#     emms_mole.to("mole / yr")

#     emms_ppb = (emms_mole / atmosphere_mole) * UNIT_REGISTRY.Quantity(1e9, "ppb")

#     return emms_ppb.to("ppb / yr")
```

```{code-cell} ipython3
# def convert_emissions(i,gases, weights, atmosphere_mole_outer):
#     """
#     i: scmdata.ScmRun
#     gases: list
#     weights: list
    
#     Return scmdata with emissions in ppb / yr
    
#     """
    

#     temp = i.timeseries().copy()
#     for gas,weight in zip(gases,weights):
#         temp.filter(variable=gas)= weight / atmosphere_mole_outer * 1e9 * 1e6*temp.filter(variable=gas).values()
#         temp.filter(variable=gas).unit = "ppb / yr"
    
#     return scmdata.ScmRun(temp)

    
# def mt_to_ppb_yr(
#     emms,
#     molar_mass,
#     atmosphere_mole=atmosphere_mole_outer,
# ):
#     emms_mole = emms / molar_mass
#     emms_mole

#     emms_ppb = (emms_mole / atmosphere_mole) *1e9

#     return emms_ppb
```

## Import Patterson emissions

```{code-cell} ipython3
path_patt = 'datasets/baseline_h2_emissions_regions.csv'

emissions_patt = pd.read_csv(path_patt).drop(columns=[
    "sector_short"])
emissions_patt
emissions_patt = emissions_patt.loc[emissions_patt["region"]=="World"].groupby(["model","region","scenario","type","unit","variable"]).sum()

patterson=scmdata.ScmRun(emissions_patt).drop_meta("type")
# patterson=scmdata.ScmRun(emissions_patt)
patterson.timeseries()
```

### Complete Past  Hydrogen emissions

```{code-cell} ipython3
def complete_hydrogen(scen,years,value):
    out= scen.timeseries().copy()    
    out[years]=value
    return scmdata.ScmRun(out)


dt_years=concentrations.filter(year=range(2016,2023))["time"]
h_avg = patterson.filter(year=range(2010,2015)).values.mean()
patterson_complete = complete_hydrogen(patterson,dt_years,h_avg)
patterson_complete.timeseries()
                                                        
```

```{code-cell} ipython3
# patterson_complete.line_plot()
```

## Import CH4 and CO emissions

```{code-cell} ipython3
path= "datasets/rcmip-emissions-annual-means-v5-1-0.csv"
# path= "datasets/SSP_CMIP6_201811.csv"

rcmip = scmdata.ScmRun(path, lowercase_cols=True).filter(region="World").filter(variable=["*|CH4","*|CO"])
```

```{code-cell} ipython3
ssp_245=rcmip.filter(scenario="ssp245")
# years=np.arange(1994,2022)
smooth_245 = ssp_245.interpolate(years)
```

```{code-cell} ipython3
smooth_245.lineplot(style='variable',marker=True)
```

```{code-cell} ipython3
smooth_245.timeseries()
```

```{code-cell} ipython3
# path= "datasets/rcmip-emissions-annual-means-v5-1-0.csv"
# rcmip = scmdata.ScmRun(path, lowercase_cols=True).filter(scenario="historical",region="World").filter(variable=["*|CH4","*|CO"])

# emissions=patterson.append(rcmip).drop_meta(["activity_id","mip_era"])
# emissions=patterson.append(smooth_245).drop_meta(["activity_id","mip_era"])
emissions=patterson_complete.append(smooth_245).drop_meta(["activity_id","mip_era"])

emissions["model"]="None"
emissions["scenario"]="historical"


emissions = emissions.filter(year=years)
emissions.timeseries()
```

### Add natural emissions

```{code-cell} ipython3
def add_natural_emissions(scen, gases, natural):
    return_emissions = scmdata.ScmRun()

    for gas,emission in zip(gases,natural):
        return_emissions = return_emissions.append(emissions.filter(variable=gas)+emission)
    
    return return_emissions
```

```{code-cell} ipython3
gases = ["Emissions|CH4","Emissions|CO","Emissions|H2"]
### natural emissions
natural = [0, 381,32.4]
combined_emissions = add_natural_emissions(emissions,gases,natural)
```

```{code-cell} ipython3
combined_emissions.timeseries()
```

```{code-cell} ipython3
combined_emissions.timeseries()
```

### Convert Emissions

```{code-cell} ipython3
# emissions
# atmosphere_molar_mass = UNIT_REGISTRY.Quantity(28.97, "g / mol")  # https://www.cs.mcgill.ca/~rwest/wikispeedia/wpcd/wp/e/Earth%2527s_atmosphere.htm#:~:text=The%20mean%20molar%20mass%20of%20air%20is%2028.97%20g%2Fmol.
# atmosphere_mass = UNIT_REGISTRY.Quantity(5.18e18, "kg") 
# atmosphere_mole_outer = (atmosphere_mass / atmosphere_molar_mass)
# atmosphere_mole_outer

molar_weights = [
    UNIT_REGISTRY.Quantity(16, "g CH4/ mol"),
    UNIT_REGISTRY.Quantity(28, "g CO/ mol"),
    UNIT_REGISTRY.Quantity(2, "g H2/ mol"),
]



emissions_ppb = scmdata.ScmRun()
for gas,weight in zip(gases,molar_weights):
    emissions_ppb = emissions_ppb.append(combined_emissions.filter(variable=gas)
                                         /weight
                                         /atmosphere_mole_outer
                                          * UNIT_REGISTRY.Quantity(1e9,"ppb"))
                                         
emissions_ppb=emissions_ppb.convert_unit("ppb/a")

emissions_ppb.timeseries()
```

```{code-cell} ipython3
emissions_ppb.timeseries()
```

### Add OH emissions

```{code-cell} ipython3
# oh_vals = 1440 * np.ones(years.shape)[np.newaxis, :]
# # append OH emissions


oh_vals = 1440 * np.ones(years.shape)[np.newaxis, :]
# add oh rate
#          Water,  nox,   o3,  tropical widening, temperature
oh_rate = (0.44  + 0.25 + 0.13 + 0.12 - 0.02)/100
rate_start = 1980-years[0]
oh_adjusted= np.arange(rate_start, years[-1]-years[0]+1)-rate_start
oh_adjusted = np.concatenate([np.zeros(rate_start),oh_adjusted* oh_rate/10])
oh_vals= oh_vals * (oh_adjusted+1)


oh_scen = scmdata.ScmRun(
         pd.DataFrame(
                oh_vals,
                index=pd.MultiIndex.from_arrays(
                    [
                        [
                           "Emissions|OH",
                           
                        ],
                        [
                            "ppb/a",
                         
                        ],
                        [
                           
                            "World",
                         
                        ],
                        [
                            "None",
                       
                        ],
                        [
                            "historical",
                         
                        ],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                ),
                columns=years,
            )
)

emissions_ppb = emissions_ppb.append(oh_scen)
for vdf in emissions_ppb.groupby("variable"):
    vdf.lineplot(style="variable")
    plt.show()
```

## Add OH concentration

```{code-cell} ipython3
# def import_rigby(path,start,end):
#     path = 'datasets/pnas.1616426114.sd01.xls'
#     rigby =pd.read_excel(path,header=4)
#     rigby_val=np.array([rigby["Unnamed: 2"].values])
#     rigby_year= np.arange(start,end+1)
#     rigby_oh= scmdata.ScmRun(
#             pd.DataFrame(
#                 rigby_val,
#                 index=pd.MultiIndex.from_arrays(
#                     [
#                         [ "Atmospheric Concentrations|OH"],
#                         ["1e6/ cm ** 3"],
#                         [
#                             "World",
#                         ],
#                         [
#                             "None",
#                         ],
#                         [
#                             "historical",
#                         ],
#                     ],
#                     names=["variable", "unit", "region", "model", "scenario"],
#                 ),
#                 columns=rigby_year,
#             )
#         )
#     return rigby_oh
    

# path = 'datasets/pnas.1616426114.sd01.xls'
# rigby = import_rigby(path, start=1980,end=2014)
# air_number = UNIT_REGISTRY.Quantity(2.5e19, "1 / cm^3")
# # air_number = UNIT_REGISTRY.Quantity(1.57e19, "1 / cm^3") # https://www.nature.com/articles/s41467-022-35419-7/tables/1
# # latitude factor
# # lat_factor = 0.48
# # rigby= rigby * lat_factor
# #Convert units and correct timerange
# rigby= rigby*UNIT_REGISTRY.Quantity(1e9, "ppb")/ air_number
# rigby =rigby.filter(year=years)
# concentrations=concentrations.append(rigby)

# rigby.timeseries()
```

```{code-cell} ipython3
concentrations.timeseries()
```

```{code-cell} ipython3
for vdf in concentrations.groupby("variable"):
    vdf.lineplot(style="variable")
    plt.show()
```

```{code-cell} ipython3
# new= new*UNIT_REGISTRY.Quantity(1e9, "ppb")/ air_number
```

```{code-cell} ipython3
# tropospheric_mass = UNIT_REGISTRY.Quantity(4.22e18,'kg').to('g')
# atmosphere_molar_mass = UNIT_REGISTRY.Quantity(28.97, "g / mol")  # https://www.cs.mcgill.ca/~rwest/wikispeedia/wpcd/wp/e/Earth%2527s_atmosphere.htm#:~:text=The%20mean%20molar%20mass%20of%20air%20is%2028.97%20g%2Fmol.
# tropospheric_mole=tropospheric_mass/atmosphere_molar_mass
# tropospheric_mole
```

```{code-cell} ipython3
# atmospheric_density = UNIT_REGISTRY.Quantity(0.0012, "g/cm**3")
# tropospheric_volume = UNIT_REGISTRY.Quantity(6e9,"km**3")
# tropospheric_volume.to("cm **3")
```

```{code-cell} ipython3
# air_number = UNIT_REGISTRY.Quantity(2.5e19, "1 / cm^3")
# air_number
```

```{code-cell} ipython3
oh_years = years.copy()

oh_concentrations= scmdata.ScmRun(
    pd.DataFrame(
        2.9e-5 * np.ones(years.shape)[np.newaxis, :],
        index=pd.MultiIndex.from_arrays(
            [
                [ "Atmospheric Concentrations|OH"],
                ["ppb"],
                [
                    "World",
                ],
                [
                    "None",
                ],
                [
                    "historical",
                ],
            ],
            names=["variable", "unit", "region", "model", "scenario"],
        ),
        columns=years,
    )
)

for vdf in oh_concentrations.groupby("variable"):
    vdf.lineplot(style="variable")
    plt.show()
    
concentrations=concentrations.append(oh_concentrations)
concentrations.timeseries()
```

```{code-cell} ipython3
# oh_years = years.copy()

# oh_concentrations= scmdata.ScmRun(
#     pd.DataFrame(
#         2e-5 * np.ones(years.shape)[np.newaxis, :],
#         index=pd.MultiIndex.from_arrays(
#             [
#                 [ "Atmospheric Concentrations|OH"],
#                 ["ppb"],
#                 [
#                     "World",
#                 ],
#                 [
#                     "None",
#                 ],
#                 [
#                     "historical",
#                 ],
#             ],
#             names=["variable", "unit", "region", "model", "scenario"],
#         ),
#         columns=years,
#     )
# )

# for vdf in oh_concentrations.groupby("variable"):
#     vdf.lineplot(style="variable")
#     plt.show()
    
# concentrations=concentrations.append(oh_concentrations)
# concentrations.timeseries()
```

```{code-cell} ipython3

```

<!-- # Experiment -->

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
emissions_ppb
```

```{code-cell} ipython3

atmosphere_mole_outer
```

```{code-cell} ipython3
# temp_nat_ch4 = UNIT_REGISTRY.Quantity(-500, "Mt CH4/ year")
# molar_mass = UNIT_REGISTRY.Quantity(16, "g CH4/ mol")
# nat_ch4_mol = temp_nat_ch4 / molar_mass
# nat_ch4_ppb = (nat_ch4_mol/atmosphere_mole_outer*UNIT_REGISTRY.Quantity(1e9,'ppb')).to('ppb / year').magnitude

# new_emissions = emissions_ppb +nat_ch4_ppb
# new_emissions.timeseries()
```

```{code-cell} ipython3
# emissions_ppb.timeseries().loc[emissions_ppb.timeseries().index.get_level_values("variable").isin(["Emissions|CH4"])]+=50
# emissions_ppb.timeseries()
```

```{code-cell} ipython3
def add_nat_ch4(i,nat_ch4):
    molar_mass = UNIT_REGISTRY.Quantity(16, "g CH4/ mol")
    out=i.timeseries().copy()
    nat_ch4_mol = nat_ch4 / molar_mass
    nat_ch4_ppb = (nat_ch4_mol/atmosphere_mole_outer*UNIT_REGISTRY.Quantity(1e9,'ppb')).to('ppb/year').magnitude
    out.loc[out.index.get_level_values("variable").isin(["Emissions|CH4"])] += nat_ch4_ppb
    return scmdata.ScmRun(out)

scen= add_nat_ch4(emissions_ppb,UNIT_REGISTRY.Quantity(-5000,"Mt CH4/ year"))
scen.timeseries()
```

```{code-cell} ipython3
def do_experiments(k1,k2, k3,kx, tau_dep_h2,alpha, hydroxyl_scale,y0_co,y0_oh,nat_ch4, input_emms, concentrations,years, y0
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
        
    emissions : scmdata.run.ScmRun
        historical emissions
        
    concentrations : scmdata.run.ScmRun
        historical concentrations
        
    years : list
        list of years which span emissions and concentrations, must be indentical
    

    
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
        
        ks: pint.Quantity[float] = field(validator=check_units(f"1 / {TIME_UNIT}"))
        """Rate constant for reaction of hydroxyl radical and everything else [1 / s]"""
        
        tau_dep_h2: pint.Quantity[float] = field(validator=check_units(TIME_UNIT))
        """Partial lifetime of hydrogen due to biogenic soil sinks [s]"""

        air_number: pint.Quantity[float] = field(
            validator=check_units(f"1 / {LENGTH_UNIT}^3"),
            default=UNIT_REGISTRY.Quantity(2.5e19, "1 / cm^3"),
        )
        """Density of air [1 / cm^3]"""
        
        alpha: pint.Quantity[float] = field(
            validator=check_units(f"1"),
            default = UNIT_REGISTRY.Quantity(0.37, ''),
        )
        """Proportion of Hydrogen produced from Methane decomposition [dimensionless]"""


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
            ks_mag = self.ks.to(f"1 / {TIME_UNIT}").magnitude
            alpha_mag = self.alpha.magnitude
            
            def dconc_dt(t, y):
                concs = {k: y[i] for k, i in _CONC_INDEXES.items()}

                emms_ch4_mag, emms_h2_mag, emms_co_mag, emms_oh_mag = emms_func(t)

                dch4dt = emms_ch4_mag - k1_mag * concs["oh"] * concs["ch4"] - ks_mag * concs["ch4"]
                dh2dt = (
                    emms_h2_mag
                    - k2_mag * concs["oh"] * concs["h2"]
                    - concs["h2"] / tau_dep_mag
                    + alpha_mag * k1_mag * concs["oh"] * concs["ch4"]
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
                    [-k1_mag * concs["oh"]-ks_mag, 0, 0, -k1_mag * concs["ch4"]],
                    [alpha_mag * k1_mag * concs["oh"],
                     -k2_mag * concs["oh"] - 1 / tau_dep_mag ,
                     0,
                     -k2_mag * concs["h2"] + alpha_mag * k1_mag * concs["ch4"]
                    ],
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
        
    def scale_hydroxyl(i,scale):
        out=i.timeseries().copy()
        out.loc[out.index.get_level_values("variable").isin(["Emissions|OH"])]*= scale
        return scmdata.ScmRun(out)

    
    def add_nat_ch4(i,nat_ch4):
        molar_mass = UNIT_REGISTRY.Quantity(16, "g CH4/ mol")
        out=i.timeseries().copy()
        nat_ch4_mol = nat_ch4 / molar_mass
        nat_ch4_ppb = (nat_ch4_mol/atmosphere_mole_outer*UNIT_REGISTRY.Quantity(1e9,'ppb')).to('ppb/year').magnitude
        out.loc[out.index.get_level_values("variable").isin(["Emissions|CH4"])] += nat_ch4_ppb
        return scmdata.ScmRun(out)

    years = years


#     y0 = {
#         "ch4": UNIT_REGISTRY.Quantity(1851.71, "ppb"),
#         "co": UNIT_REGISTRY.Quantity(132.4, "ppb"),
#         "h2": UNIT_REGISTRY.Quantity(432.33, "ppb"),
#         "oh": UNIT_REGISTRY.Quantity(2.22e-05, "ppb"),
#     } 
#     y0 = {
#         "ch4": UNIT_REGISTRY.Quantity(1680, "ppb"),
#         "co": UNIT_REGISTRY.Quantity(60, "ppb"),
#         "h2": UNIT_REGISTRY.Quantity(530, "ppb"),
#         "oh": UNIT_REGISTRY.Quantity(2.48e-05, "ppb"),
#     } 
    # scale hydroxyl
    input_emms = scale_hydroxyl(input_emms,hydroxyl_scale.magnitude)
    
    # Add natural methane

    # add exta natural ch4 emissions
    input_emms = add_nat_ch4(input_emms,nat_ch4)

    # modify first co and oh values
    y0["co"]=y0_co
    y0["oh"]=y0_oh
    
    
    to_solve = HydrogenBox(
    k1=k1,
    k2=k2,
    k3=k3,
    kx=kx,
    ks=UNIT_REGISTRY.Quantity(0.02,"1 / year"),
    tau_dep_h2=tau_dep_h2,
    alpha=alpha
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
        ),):
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

#     out = scmdata.run_append(scens_res).filter()
    # drop OH
    out = scmdata.run_append(scens_res).filter(variable=["*|CH4","*|CO","*|H2"])
    return out
```

```{code-cell} ipython3
# target.timeseries()
```

```{code-cell} ipython3
# target
```

### Target

Use Jpl values to calculate a target.

```{code-cell} ipython3
k1_jpl = UNIT_REGISTRY.Quantity(6.3e-15, "cm^3 / s")
k2_jpl = UNIT_REGISTRY.Quantity(6.7e-15, "cm^3 / s")  # JPL publication 19-5, page 1-53
k3_jpl = UNIT_REGISTRY.Quantity(2e-13, "cm^3 / s")
kx_base = UNIT_REGISTRY.Quantity(1.062, "1 / s")
```

```{code-cell} ipython3
emissions_ppb
```

```{code-cell} ipython3
concentrations.timeseries()
```

```{code-cell} ipython3
# y0 = {
#         "ch4": UNIT_REGISTRY.Quantity(974.787109, "ppb"),
#         "co": UNIT_REGISTRY.Quantity(100, "ppb"),
#         "h2": UNIT_REGISTRY.Quantity(342.044854, "ppb"),
#         "oh": UNIT_REGISTRY.Quantity(2.9e-05, "ppb"),
#     } 
```

```{code-cell} ipython3
concentrations
```

```{code-cell} ipython3
years_co[0]
```

```{code-cell} ipython3
def get_y0(conc):
    '''
    returns correct format for y0
    Lettercase of gas not important for solve function
    
    '''
    gases = ["CH4","H2","OH"]
    y0 = {gas.lower() : UNIT_REGISTRY.Quantity(conc.filter(variable = '*|'+gas).timeseries().iloc[0,0], 'ppb')for gas in gases}
    return y0

y0=get_y0(concentrations)
truth = {
    "k1" : UNIT_REGISTRY.Quantity(6.3e-15, "cm^3 / s"),
    "k2" : UNIT_REGISTRY.Quantity(6.7e-15, "cm^3 / s") ,
    "k3" : UNIT_REGISTRY.Quantity(2e-13, "cm^3 / s"),
    "kx" : UNIT_REGISTRY.Quantity(0.8, "1/ s"),
    "tau_dep_h2" : UNIT_REGISTRY.Quantity(2.63, "year"),
    "alpha" : UNIT_REGISTRY.Quantity(0.32,"1"),
    "hydroxyl_scale": UNIT_REGISTRY.Quantity(1,""),
    "y0_co":UNIT_REGISTRY.Quantity(70,"ppb"),
    "y0_oh":UNIT_REGISTRY.Quantity(2.5e-5,"ppb"),
    "nat_ch4":UNIT_REGISTRY.Quantity(305,"Mt CH4 / a"),
    "input_emms" : emissions_ppb,
    "concentrations" : concentrations,
    "years" : years,
    "y0" : y0,

}

# get correct format of target
target = do_experiments(**truth,)
target["model"] = "target"


target = concentrations.interpolate(target["time"])
target["model"] = "target"

# fix target timeseries.
def set_past_co_nan(scen,years):
    out=scen.copy().timeseries()
    out.loc[out.index.get_level_values("variable").isin(["Atmospheric Concentrations|CO"]),:dt.datetime(years[0],1,1)]=np.nan
    return scmdata.ScmRun(out)

target=set_past_co_nan(target,years_co)

for vdf in target.groupby("variable"):
    vdf.lineplot(style="variable",marker='.')
    plt.show()
```

```{code-cell} ipython3
target.timeseries()
```

```{code-cell} ipython3
# target[]
```

```{code-cell} ipython3

```

### Cost calculation

The next thing is to decide how we're going to calculate the cost function. There are many options here, in this case we're going to use the sum of squared errors.

```{code-cell} ipython3
# concentrations.timeseries().mean(axis=1)
```

```{code-cell} ipython3
# normalisation = pd.Series(
# #     [0.1,0.1,0.1,0.1],
#     [1,1,1],

#     index=pd.MultiIndex.from_arrays(
#         (
#             [
#                 "Atmospheric Concentrations|CH4",
#                 "Atmospheric Concentrations|H2",
#                 "Atmospheric Concentrations|CO",
#             ]
#               ,
#             ["ppb","ppb","ppb"],
#         ),
#         names=["variable", "unit"],
#     ),
# )
# normalisation
```

## normalisation with spinup time

```{code-cell} ipython3
np.array(1)/np.array(np.nan)
```

```{code-cell} ipython3
np.nansum(concentrations.values)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
normalisation_values= concentrations.timeseries().mean(axis=1).values
normalisation_names=[gas for gas in concentrations["variable"]]
np.ones(years.shape)
normalisation_series=normalisation_values[:,np.newaxis]* np.ones(years.shape)[np.newaxis, :]

# spinup = 1 #
# normalisation_series[:,0:spinup]=1e40

normalisation_series
normalisation_years=target["time"].values
normalisation_scen = scmdata.ScmRun(
         pd.DataFrame(
                normalisation_series,
                index=pd.MultiIndex.from_arrays(
                    [
                        ["Atmospheric Concentrations|CH4",
                        "Atmospheric Concentrations|CO",
                        "Atmospheric Concentrations|H2",
                        "Atmospheric Concentrations|OH",
                        ],
                        ["ppb","ppb","ppb","ppb"],
                        ["World","World","World","World",],
                        ["target","target","target","target",],
                        ["historical","historical","historical","historical",],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                    ),
                    columns=normalisation_years
         )
)
```

```{code-cell} ipython3
# normalisation_scen=set_past_co_nan(normalisation_scen,years_co)
normalisation_scen.timeseries()
```

```{code-cell} ipython3
# turn Hydroxyl off

target= target.filter(variable=["*|CH4","*|CO","*|H2"])
normalisation_scen=normalisation_scen.filter(variable=["*|CH4","*|CO","*|H2"])
                    
```

```{code-cell} ipython3
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    import attr
    import pandas as pd
    import scmdata.run


def _works_with_self_target(
    instance: OptCostCalculatorSSE,
    attribute: attr.Attribute[scmdata.run.BaseScmRun],
    value: scmdata.run.BaseScmRun,
) -> None:
    def _get_msg() -> str:
        target_ts = instance.target.timeseries()
        value_ts = value.timeseries()

        msg = (
            "target and normalisation are somehow misaligned "
            "(passing self.target to self.calculate_cost results "
            "in nan), please check.\n"
            f"target timeseries:\n{target_ts}\n"
            f"{attribute.name} timeseries:\n{value_ts}"
        )

        return msg

    try:
        instance.calculate_cost(instance.target)
    except KeyError as exc:
        raise ValueError(_get_msg()) from exc


def _is_meta_in_target(
    instance: OptCostCalculatorSSE,
    attribute: attr.Attribute[str],
    value: str,
) -> None:
    available_metadata = instance.target.meta_attributes
    if value not in available_metadata:
        msg = (
            f"value of ``{attribute.name}``, '{value}' is not in the metadata "
            f"of target. Available metadata: {available_metadata}"
        )

        raise KeyError(msg)




@define
class MyCostCalculatorSSE:  # pylint: disable=too-few-public-methods
    """
    Cost calculator based on sum of squared errors

    This is a convenience class. We may want to refactor it in future to
    provide greater flexibility for other cost calculations.
    """

    target: scmdata.run.BaseScmRun
    """Target timeseries"""

    model_col: str = field(validator=[_is_meta_in_target])
    """
    Column which contains the name of the model.

    This is used when subtracting the model results from the target
    """

    normalisation: scmdata.run.BaseScmRun = field(validator=[_works_with_self_target])
    """
    Normalisation values

    Should have same timeseries as target. See the class methods for helpers.
    """

    @classmethod
    def from_unit_normalisation(
        cls, target: scmdata.run.BaseScmRun, model_col: str
    ) -> OptCostCalculatorSSE:
        """
        Initialise assuming unit normalisation for each timeseries.

        This is a convenience method, but is not recommended for any serious
        work as unit normalisation is unlikely to be a good choice for most
        problems.

        Parameters
        ----------
        target
            Target timeseries

        model_col
            Column which contains of the model in ``target``

        Returns
        -------
            :obj:`OptCostCalculatorSSE` such that the normalisation is 1 for
            all timepoints (with the units defined by whatever the units of
            each timeseries are in ``target``)
        """
        norm = target.timeseries()
        norm.loc[:, :] = 1
        norm = type(target)(norm)

        return cls(target=target, normalisation=norm, model_col=model_col)


    @classmethod
    def from_series_normalisation(
        cls,
        target: scmdata.run.BaseScmRun,
        model_col: str,
        normalisation_series: pd.Series,
    ) -> OptCostCalculatorSSE:
        """
        Initialise starting from a series that defines normalisation for each timeseries.

        The series is broadcast to match the timeseries in target, using the
        same value for all timepoints in each timeseries.

        Parameters
        ----------
        target
            Target timeseries

        model_col
            Column which contains of the model in ``target``

        normalisation_series
            Series to broadcast to create the desired normalisation

        Returns
        -------
            Initialised :obj:`OptCostCalculatorSSE`
        """
        required_columns = {"variable", "unit"}
        missing_cols = required_columns - set(normalisation_series.index.names)
        if missing_cols:
            msg = (
                "normalisation is missing required column(s): "
                f"``{sorted(missing_cols)}``"
            )
            raise KeyError(msg)

        target_ts_no_unit = target.timeseries().reset_index("unit", drop=True)

        # This is basically what pandas does internally when doing ops:
        # align and then broadcast
        norm_series_aligned, _ = normalisation_series.align(target_ts_no_unit)

        if norm_series_aligned.isnull().any().any():
            msg = (
                "Even after aligning, there are still nan values.\n"
                f"norm_series_aligned:\n{norm_series_aligned}\n"
                f"target_ts_no_unit:\n{target_ts_no_unit}"
            )
            raise ValueError(msg)

        if norm_series_aligned.shape[0] != target_ts_no_unit.shape[0]:
            msg = (
                "After aligning, there are more rows in the normalisation "
                "than in the target.\n"
                f"norm_series_aligned:\n{norm_series_aligned}\n"
                f"target_ts_no_unit:\n{target_ts_no_unit}"
            )
            raise ValueError(msg)

        norm_series_aligned = type(target_ts_no_unit)(
            np.broadcast_to(norm_series_aligned.values, target_ts_no_unit.T.shape).T,
            index=norm_series_aligned.index,
            columns=target_ts_no_unit.columns,
        )

        normalisation = type(target)(norm_series_aligned)

        return cls(target=target, normalisation=normalisation, model_col=model_col)


    def calculate_cost(self, model_results: scmdata.run.BaseScmRun) -> float:
        """
        Calculate cost function based on model results

        Parameters
        ----------
        model_results
            Model results of which to calculate the cost

        Returns
        -------
            Cost
        """
        diff = model_results.subtract(
            self.target, op_cols={self.model_col: "res - target"}
        ).divide(
            self.normalisation,
            op_cols={self.model_col: "(res - target) / normalisation"},
        )

        cost = float(np.nansum(np.nansum((diff.convert_unit("1") ** 2).values)))

        return cost
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
cost_calculator = MyCostCalculatorSSE(
    target=target,  model_col="model",normalisation=normalisation_scen,
)
```

```{code-cell} ipython3
cost_calculator.normalisation.timeseries()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# normalisation = pd.Series(
# #     [0.1,0.1,0.1,0.1],
#     [1,1,1,1e-9],

#     index=pd.MultiIndex.from_arrays(
#         (
#             [
#                 "Atmospheric Concentrations|CH4",
#                 "Atmospheric Concentrations|CO",
#                 "Atmospheric Concentrations|H2",
#                 "Atmospheric Concentrations|OH",
#             ],
#             ["ppb","ppb","ppb","ppb"],
#         ),
#         names=["variable", "unit"],
#     ),
# )
# normalisation
```

```{code-cell} ipython3
# normalisation_values= concentrations.timeseries().mean(axis=1).values
# normalisation_names=[gas for gas in concentrations["variable"]]

# #Reduce OH normalisation
# normalisation_values[3]=1e30
# # normalisation_values[2]=1e-5


# normalisation = pd.Series(
#     normalisation_values,

#     index=pd.MultiIndex.from_arrays(
#         (
#             normalisation_names,
#             ["ppb","ppb","ppb","ppb"],
#         ),
#         names=["variable", "unit"],
#     ),
# )
# normalisation


# # normalisation = pd.Series(
# # #     [0.1,0.1,0.1,0.1],
# #     [1,1,1,1e-9],

# #     index=pd.MultiIndex.from_arrays(
# #         (
# #             [
# #                 "Atmospheric Concentrations|CH4",
# #                 "Atmospheric Concentrations|CO",
# #                 "Atmospheric Concentrations|H2",
# #                 "Atmospheric Concentrations|OH",
# #             ],
# #             ["ppb","ppb","ppb","ppb"],
# #         ),
# #         names=["variable", "unit"],
# #     ),
# # )
# # normalisation
```

#

```{code-cell} ipython3
# cost_calculator = OptCostCalculatorSSE.from_series_normalisation(
#     target=target, normalisation_series=normalisation, model_col="model"
# )

assert cost_calculator.calculate_cost(target) == 0
assert cost_calculator.calculate_cost(target * 1.1) > 0
cost_calculator
```

```{code-cell} ipython3
cost_calculator
```

```{code-cell} ipython3
cost_calculator
```

```{code-cell} ipython3
cost_calculator.calculate_cost(target*1.2) 
```

```{code-cell} ipython3
cost_calculator.normalisation.timeseries()
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
    ("kx", f" 1 / {TIME_UNIT}"),
    ("tau_dep_h2", f"year"),
    ("alpha", f""),
    ("hydroxyl_scale",f''),
    ('y0_co',f'{CONC_UNIT}'),
    ('y0_oh',f'{CONC_UNIT}'),
    ('nat_ch4',f'Mt CH4 / year'),
]
parameters
```

Next we define a function which, given pint quantities, returns the inputs needed for our `do_experiments` function. In this case this is not a very interesting function, but in other use cases the flexibility is helpful.

```{code-cell} ipython3
def do_model_runs_input_generator(
    k1: pint.Quantity, k2: pint.Quantity, k3: pint.Quantity, kx: pint.Quantity,
    tau_dep_h2: pint.Quantity, alpha:pint.Quantity, hydroxyl_scale: pint.Quantity, 
    y0_co: pint.Quantity, y0_oh: pint.Quantity, nat_ch4: pint.Quantity,
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
        
  

    Returns
    -------
        Inputs for :func: do_experiments
    """
    return {"k1": k1, "k2": k2, "k3": k3, "kx":kx, "tau_dep_h2": tau_dep_h2, 
            "alpha": alpha, "hydroxyl_scale": hydroxyl_scale, "y0_co":y0_co, "y0_oh":y0_oh,
            "input_emms" : emissions_ppb, "concentrations" : concentrations,
            "years" : years, "y0": y0, "nat_ch4":nat_ch4,}
```

```{code-cell} ipython3
model_runner = OptModelRunner.from_parameters(
    params=parameters,
    do_model_runs_input_generator=do_model_runs_input_generator,
    do_model_runs=do_experiments,
)
```

Now we can run from a plain numpy array (like scipy will use) and get a result that will be understood by our cost calculator.

```{code-cell} ipython3
cost_calculator.calculate_cost(model_runner.run_model([5e-15, 5e-15, 2.3e-13, 0.7, 2.3,0.2,1,50,2.5e-5,305]))
```

We have to define where to start the optimisation.

```{code-cell} ipython3
start = np.array([5e-15, 5e-15, 2.3e-13, 0.7, 2.3,0.2,1,50,2.5e-5,305])
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
# bounds_dict = {
#     "k1": [
#         UNIT_REGISTRY.Quantity(1e-17, "cm^3 / s"),
#         UNIT_REGISTRY.Quantity(1e-10, "cm^3 / s"),
#     ],
#     "k2": [
#         UNIT_REGISTRY.Quantity(1e-17, "cm^3 / s"),
#         UNIT_REGISTRY.Quantity(1e-14, "cm^3 / s"),
#     ],
#     "k3": [
#         UNIT_REGISTRY.Quantity(1e-17, "cm^3 / s"),
#         UNIT_REGISTRY.Quantity(1e-10, "cm^3 / s"),
#     ],

# }
bounds_dict = {
    "k1": [
        UNIT_REGISTRY.Quantity(4e-15, "cm^3 / s"),
        UNIT_REGISTRY.Quantity(9e-15, "cm^3 / s"),
    ],
    "k2": [
        UNIT_REGISTRY.Quantity(1e-15, "cm^3 / s"),
        UNIT_REGISTRY.Quantity(1e-14, "cm^3 / s"),
    ],
    "k3": [
        UNIT_REGISTRY.Quantity(2.2e-13, "cm^3 / s"),
        UNIT_REGISTRY.Quantity(2.4e-13, "cm^3 / s"),
    ],
    "kx": [
        UNIT_REGISTRY.Quantity(0.5, "1/ s"),
        UNIT_REGISTRY.Quantity(1.1, "1 / s"),
    ],
     "tau_dep_h2": [
        UNIT_REGISTRY.Quantity(2, " year"),
        UNIT_REGISTRY.Quantity(3, " year"),
    ],
    "alpha": [
        UNIT_REGISTRY.Quantity(0, ""),
        UNIT_REGISTRY.Quantity(0.4, ""),
    ],
    "hydroxyl_scale": [
        UNIT_REGISTRY.Quantity(0.9,""),
        UNIT_REGISTRY.Quantity(1.1,"")
    ],
     "y0_co":[
        UNIT_REGISTRY.Quantity(40,"ppb"),
        UNIT_REGISTRY.Quantity(80,"ppb")
    ],
    "y0_oh":[
        UNIT_REGISTRY.Quantity(1.9e-5,"ppb"),
        UNIT_REGISTRY.Quantity(3e-5,"ppb")
    ],
    "nat_ch4":[
        UNIT_REGISTRY.Quantity(0,"Mt CH4 / year"),
        UNIT_REGISTRY.Quantity(500,"Mt CH4 / year")
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
atol = 0
tol = 0.0002
# Maximum number of iterations to use
maxiter = 36
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
thin_ts_to_plot = 3


# Create axes to plot on (could also be created as part of a factory
# or class method)
convert_scmrun_to_plot_dict = partial(scmrun_as_dict, groups=["variable", "scenario"])

cost_name = "cost"
timeseries_axes = list(convert_scmrun_to_plot_dict(target).keys())

parameters_names = [v[0] for v in parameters]
parameters_mosaic = list(more_itertools.repeat_each(parameters_names, 1))
timeseries_axes_mosaic = list(more_itertools.repeat_each(timeseries_axes, 1))

# if len(parameters_mosaic)!=len(timeseries_axes_mosaic):
#     while len(parameters_mosaic)<len(timeseries_axes_mosaic):
#         parameters_mosaic.append(".")
#     while len(parameters_mosaic)>len(timeseries_axes_mosaic):
#         timeseries_axes_mosaic.append(".")

fig, axd = plt.subplot_mosaic(
    mosaic=[
        [cost_name]+timeseries_axes_mosaic,
#         [timeseries_axes_mosaic[0]]+timeseries_axes_mosaic,
        [cost_name]+parameters_mosaic[0:3],
        [cost_name]+parameters_mosaic[3:-4],
        [cost_name]+parameters_mosaic[-4:-1],
        [cost_name]+[parameters_mosaic[-1]]*3,


    ],
    figsize=(12, 12),
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

```{code-cell} ipython3
UNIT_REGISTRY.Quantity(9.37e7,"s").to("year")
```

```{code-cell} ipython3
raise SystemExit("Stop right there!")
```

## Local optimisation

Scipy also has [local optimisation](https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization) (e.g. Nelder-Mead) options. Here we show how to do this.

+++

Again, we have to define where to start the optimisation (this has a greater effect on local optimisation).

```{code-cell} ipython3
# Here we imagine that we're polishing from the results of the DE above,
# but we make the start slightly worse first
start_local = optimize_res.x
start_local
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# Optimisation parameters
tol = 1e-3
# Maximum number of iterations to use
maxiter = 128

# I think this is how this works
max_n_runs = len(parameters) + 5 * maxiter

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
            n_timeseries_per_row=2,
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
        bounds=bounds
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
raise SystemExit("Stop right there!")
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
max_iterations = 500
burnin =100
thin = 2

## Visualisation options
plot_every = 15
convergence_ratio = 5
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

```{code-cell} ipython3
op
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{raw-cell}
# 
```
