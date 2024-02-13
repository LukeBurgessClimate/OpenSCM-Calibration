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

## Read rcmip data

```{code-cell} ipython3

```

```{code-cell} ipython3
import numpy as np
import pandas as pd
import scmdata
import matplotlib.pyplot as plt
import pint
from openscm_units import unit_registry
import datetime as dt
```

```{code-cell} ipython3
UNIT_REGISTRY = unit_registry
# add hydrogen in here
symbol = "H2"
value = "hydrogen"
UNIT_REGISTRY.define(f"{symbol} = [{value}]")
UNIT_REGISTRY.define(f"{value} = {symbol}")
UNIT_REGISTRY._add_mass_emissions_joint_version(symbol)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
path= "datasets/rcmip-emissions-annual-means-v5-1-0.csv"
emissions = pd.read_csv(path)
```

```{code-cell} ipython3
# # emissions
# emissions[emissions["Variable"].isin(emissions_names)]
```

```{code-cell} ipython3
historical_name='historical'
emissions_names = ["Emissions|CH4","Emissions|CO"]
start = "1850"
end = "2020"
# historical_emissions=emissions.copy()[emissions["Scenario"]==historical_name]
```

```{code-cell} ipython3
rcmip
```

```{code-cell} ipython3
rcmip = scmdata.ScmRun(path, lowercase_cols=True).filter(region="World",scenario="historical").filter(variable=["*|CH4","*|CO"])
# rcmip = rcmip.drop_meta(["activity_id", "mip_era"])
# rcmip.timeseries().iloc[:,[-200]]
scens= rcmip
for vdf in scens.groupby("variable"):
    ax = vdf.lineplot(style="variable")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.grid()
    plt.show()
```

```{code-cell} ipython3
rcmip.filter(year=range(2000,2040)).lineplot()
```

```{code-cell} ipython3
# path_patt = 'datasets/baseline_h2_emissions_regions.csv'
# # patterson = scmdata.ScmRun(path_patt, lowercase_cols=True).drop_meta("sector_short")
# patterson = scmdata.ScmRun(path_patt, lowercase_cols=True)
# # patterson.drop_meta("sector_short")
# patterson
```

```{code-cell} ipython3
path_patt = 'datasets/baseline_h2_emissions_regions.csv'

emissions_patt = pd.read_csv(path_patt).drop(columns=[
    "sector_short"])
emissions_patt
emissions_patt = emissions_patt.loc[emissions_patt["region"]=="World"].groupby(["model","region","scenario","type","unit","variable"]).sum()
# patterson=scmdata.ScmRun(emissions_patt).drop_meta("type")
patterson=scmdata.ScmRun(emissions_patt)
type(patterson)
```

```{code-cell} ipython3
patterson_ppb = patterson.filter(variable="Emissions|H2").timeseries().copy()

patterson_ppb.loc[patterson_ppb.index.get_level_values("variable")
                 =="Emissions|H2",
                  :]*UNIT_REGISTRY(2.12, "ppb / (Mt H2)")
```

```{code-cell} ipython3
combined=patterson.append(rcmip)
combined["]
```

```{code-cell} ipython3
palette = scmdata.plotting.RCMIP_SCENARIO_COLOURS
palette
palette["Emissions|CH4"] = "#003466"
palette["Emissions|CO"] = "#709fcc"
palette["Emissions|CO"] = "#709fcc"
```

```{code-cell} ipython3
var_to_plot = "Emissions|CH4"
ylim = [-10, 1200]
# var_to_plot = "Emissions|CO2|*AFOLU*" # ylim = [-3, 3]
pdf = rcmip.filter(
     year=range(1, 2501)
)
pkwargs = dict(
hue="variable", legend=True, palette=palette, linewidth=3, alpha=0.7
)
fig = plt.figure(figsize=(16, 18))
ax = fig.add_subplot(111)
pdf.lineplot(
    ax=ax, **pkwargs, hue_order=sorted(pdf.get_unique_meta("variable"))
)
ax.legend(loc="center right")
ax.set_ylim(ylim)
```

# Calculate natural emissions

```{code-cell} ipython3
def plot_emissions(emissions,start,end,x_range,y_range):
    plt.figure()
    emissions.lineplot()
    plt.xlim([dt.datetime(x_range[0],1,1),dt.datetime(x_range[1],1,1)])
    plt.ylim(y_range)
    plt.vlines([dt.datetime(start,1,1),dt.datetime(end,1,1)],y_range[0],y_range[1],colors='k')
    
start = 1850
end  = 1900
path= "datasets/rcmip-emissions-annual-means-v5-1-0.csv"

rcmip = scmdata.ScmRun(path, lowercase_cols=True)
#CH4
ch4_run = rcmip.filter(region="World", variable = ["*|CH4"],scenario = "*historical")
ch4_hist = ch4_run.filter(year = range(start, end)).values.mean()
print(f'Average historical value for CH4 is {ch4_hist:.2f} Mt/ yr')

#CO
co_run = rcmip.filter(region="World", variable = ["*|CO"],scenario = "*historical")
co_hist = co_run.filter(year = range(start, end)).values.mean()
print(f'Average historical value for CO is {co_hist:.2f} Mt/ yr')
#h2
h2_run = patterson.copy()
h2_hist = h2_run.filter(year = range(start, end)).values.mean()
print(f'Average historical value for H2 is {h2_hist:.2f} Mt/ yr')


# rcmip.filter(region="World", variable = ["*|CH4"],scenario = "*historical").lineplot()
# plt.ylim([0,400])

# plt.figure()
# rcmip.filter(region="World", variable = ["*|CO"],scenario = "*historical").lineplot()
# plt.ylim([0,1200])

# plt.figure()
# patterson.lineplot()
# plt.ylim([0,40])
plot_emissions(rcmip.filter(region="World", variable = ["*|CH4"],scenario = "*historical"),start,end,[1750,2020],[0,400])
plot_emissions(rcmip.filter(region="World", variable = ["*|CO"],scenario = "*historical"),start,end,[1750,2020],[0,1200])

plot_emissions(patterson,start,end,[1750,2020],[0,40])
```

## Import Agage data

```{code-cell} ipython3
noaa_paths = ['datasets/ch4_cgo_surface-flask_1_ccgg_event.txt','datasets/co_cgo_surface-flask_1_ccgg_event.txt','datasets/h2_cgo_surface-flask_1_ccgg_event.txt']
paths = ['datasets/AGAGE-GCMD_CGO_ch4.txt','datasets/AGAGE-GCMD_CGO_co.txt','datasets/AGAGE-GCMD_CGO_h2.txt']

gases = ["CH4", "CO", "H2"]
headers = [162, 160,164]


def load_gas_csv(paths,gases):
    df =[]
    for path, gas in zip(paths,gases):
        daily= pd.read_csv(path, header=16,delim_whitespace=True)
        daily=daily[daily["flag"]=="B"]

        yearly = daily.groupby("YYYY")["mole_fraction"].mean()
        df.append(yearly)
    return df

def load_noaa_csv(paths,gases,headers):
    df =[]
    for path, gas, header in zip(paths,gases,headers):
        daily= pd.read_csv(path, header=header,delim_whitespace=True)
        
        #Filter data
        daily=daily[daily["qcflag"]=="..."]
        
        yearly = daily.groupby("year")["value"].mean()
        df.append(yearly)
    return df

```

```{code-cell} ipython3
#path= 'datasets/co_cgo_surface-flask_1_ccgg_event.txt'
path = 'datasets/AGAGE-GCMD_CGO_ch4.txt'
hydrogen_daily = pd.read_csv(path,header=16, delim_whitespace=True)
hydrogen_daily[hydrogen_daily['flag']=="B"]

# hydrogen_yearly = hydrogen_daily.groupby("YYYY")["mole_fraction"].mean()
# hydrogen_daily["flag"].unique()
# hydrogen_daily[hydrogen_daily['flag']=="B"]
# hydrogen_daily[]
# hydrogen_daily["mole_fraction"].plot()
hydrogen_daily[hydrogen_daily['flag']=="B"]["mole_fraction"].plot()
# hydrogen_daily[hydrogen_daily['flag']=="P"]["mole_fraction"].plot(alpha=0.5)
# 
```

```{code-cell} ipython3
path= 'datasets/h2_cgo_surface-flask_1_ccgg_event.txt'
hydrogen_daily = pd.read_csv(path,header=164, delim_whitespace=True)

# hydrogen_daily = pd.read_csv(path,header=16, delim_whitespace=True)
# hydrogen_yearly = hydrogen_daily.groupby("YYYY")["mole_fraction"].mean()
# hydrogen_yearly.index


# co_daily = pd.read_csv(paths[1],header=16, delim_whitespace=True)
# hydrogen_daily[hydrogen_daily["qcflag"]=="..."]
hydrogen_daily=hydrogen_daily[hydrogen_daily["qcflag"]=="..."]
hydrogen_daily.groupby(["year","month"]).count()
hydrogen_daily["value"].plot()
```

```{code-cell} ipython3
# hydrogen_daily.groupby(["year"]).count().head(30)
```

```{code-cell} ipython3
yearly = load_gas_csv(paths,gases)
# plt.plot(co_daily)
# co_daily[co_daily['flag']=="P"]
yearly
```

```{code-cell} ipython3
yearly2 = load_noaa_csv(noaa_paths,gases,headers)
yearly2
```

```{code-cell} ipython3
for i in range(3):
    plt.figure()
    plt.plot(yearly[i],label="agage")
    plt.plot(yearly2[i],label="noaa")
    plt.legend()
    plt.xlim([1983,2023])
    plt.title(gases[i])
```

```{code-cell} ipython3
yearly[2].loc[1994]
```

```{code-cell} ipython3
hydrogen_combine = [yearly2[0].loc[1994:2022],yearly2[1].loc[1994:2022],yearly[2].loc[1994:2022]]
hydrogen_combine
```

```{code-cell} ipython3
years=np.arange(1993,2023)
gas_scen = scmdata.ScmRun(
         pd.DataFrame(
                hydrogen_combine,
                index=pd.MultiIndex.from_arrays(
                    [
                        [
                           "CH4",
                            "CO",
                            "H2"
                        ],
                        [
                            "CH4 ppb",
                            "CO ppb",
                            "H2 ppb"
                        ],
                        [
                           
                            "CGO",
                            "CGO",
                            "CGO",
                        ],
                        [
                            "model",
                            "model",
                            "model",
                        ],
                        [
                            "historical",
                            "historical",
                            "historical",
                        
                        ],
                    ],
                    names=["variable", "unit", "region", "model", "scenario"],
                ),
                columns=years,
            )
)
for vdf in gas_scen.groupby("variable"):
    vdf.lineplot(label="variable")
    plt.show()
```

```{code-cell} ipython3
fname="datasets/historical_gas_conc.nc"
scmdata.netcdf.run_to_nc(gas_scen,fname)
```

```{code-cell} ipython3
loaded_nc = scmdata.ScmRun.from_nc(fname=fname)
loaded_nc
for vdf in loaded_nc.groupby("variable"):
    vdf.lineplot()
    plt.show()
```

```{code-cell} ipython3
years = np.arange(10,150)
pd.DataFrame(
        np.array([177, 240, 552 / 2, 1180])[:, np.newaxis]
        #         np.array([177, 552 / 2, 240, 1180])[:, np.newaxis]
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
    
    columns=years
)
```

```{code-cell} ipython3
np.array([177, 240, 552 / 2, 1180])[:, np.newaxis]
```

```{code-cell} ipython3
# hello = scmdata.ScmRun(hyd
#          pd.DataFrame(
#                 hydrogen_yearly,
#                 index=pd.MultiIndex.from_arrays(
#                     [
#                         [
#                             f"Atmospheric Concentrations|{c.upper()}"
#                             for c in _CONC_ORDER
#                         ],
#                         [CONC_UNIT, CONC_UNIT, CONC_UNIT, CONC_UNIT],
#                         [
#                             "World",
#                             "World",
#                             "World",
#                             "World",
#                         ],
#                         [model, model, model, model],
#                         [
#                             scenario,
#                             scenario,
#                             scenario,
#                             scenario,
#                         ],
#                     ],
#                     names=["variable", "unit", "region", "model", "scenario"],
#                 ),
#                 columns=out_years,
#             )
#         )
    
```

## Hydroxyl trend

+++

###  Rigby

```{code-cell} ipython3
rigby.rename(columns={"Unnamed: 0":"year"}).head()
```

```{code-cell} ipython3
rigby.tail()
```

```{code-cell} ipython3
def import_rigby(path,start,end):
    path = 'datasets/pnas.1616426114.sd01.xls'
    rigby =pd.read_excel(path,header=4)
    rigby_val=np.array([rigby["Unnamed: 2"].values])
    rigby_year= np.arange(start,end+1)
    rigby_oh= scmdata.ScmRun(
            pd.DataFrame(
                rigby_val,
                index=pd.MultiIndex.from_arrays(
                    [
                        [ "Atmospheric Concentrations|OH"],
                        ["1/ cm ** 3"],
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
                columns=rigby_year,
            )
        )
    return rigby_oh
    
start = 1980
end = 2014
path = 'datasets/pnas.1616426114.sd01.xls'

rigby = import_rigby(path, 1980,2014)
rigby.timeseries()
```

```{code-cell} ipython3
rigby
```

```{code-cell} ipython3
path = 'datasets/pnas.1616426114.sd01.xls'
rigby =pd.read_excel(path,header=4)
rigby["year"]=rigby["Unnamed: 0"].values
```

```{code-cell} ipython3
np.array([rigby["Unnamed: 2"]]).shape
```

```{code-cell} ipython3
rigby_values = pd.DataFrame(rigby["Unnamed: 2"].values,index=rigby["year"],
                            columns=["Atmospheric Concentration|OH"]
                           )
rigby_values.head()
type(rigby_values)
```

```{code-cell} ipython3
years
```

```{code-cell} ipython3
a=2.9e-5 * np.ones(years.shape)[np.newaxis, :]
a.shape
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
df = pd.DataFrame(
    [["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
    columns=["first", "second"],
)
df2 =pd.MultiIndex.from_frame(df)
```

```{code-cell} ipython3
rigby_values.transpose()[]
```

```{code-cell} ipython3
rigby_scen = scmdata.ScmRun(rigby_values.tranpose)
```

```{code-cell} ipython3
np.array(rigby_values["Atmospheric Concentration|OH"].values).transpose()
```

```{code-cell} ipython3
rigby_conc= scmdata.ScmRun(rigby_values.transpose(),
                           index=pd.MultiIndex.from_arrays(
                               [["OH"],["World"],["historical"],["10**6 / cm-3"],["None"]],
                               names=["variable", "unit", "region", "model", "scenario"]
                           ),
                            columns =np.arange(1980,2014) 
                        )
```

```{code-cell} ipython3
plt.plot(rigby_values)
```

### Naus

```{code-cell} ipython3

```

```{code-cell} ipython3
import netCDF4
```

```{code-cell} ipython3
# fname="datasets/OH_scaling_latitudinal_POP_20y.nc4"
fname="datasets/OH_scaling_latitudinal_REF_20y.nc4"

hydroxyl_nc = netCDF4.Dataset(fname,'r')
# hydroxyl_nc2 = scmdata.ScmRun.from_nc(fname=fname)
fh= netCDF4.Dataset(fname)
```

```{code-cell} ipython3
for i in fh.variables:
    print(i)
```

```{code-cell} ipython3
time = fh.variables['time'][:]
lats = fh.variables['latitude'][:]
oh_scal = fh.variables['oh_scaling'][:]

```

```{code-cell} ipython3
# oh_scal[:,12]
```

```{code-cell} ipython3
# time
```

```{code-cell} ipython3
oh_scal.shape
```

```{code-cell} ipython3
plt.plot(time[:,0],oh_scal[:,12],'.')
```

```{code-cell} ipython3
fh[-40,:,:]
```

```{code-cell} ipython3
fh.close()
```

```{code-cell} ipython3
plt.contourf(fh['oh_scaling'])
fh
```

```{code-cell} ipython3

```
