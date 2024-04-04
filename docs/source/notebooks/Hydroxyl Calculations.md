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

# Hydroxyl Calculations

+++

Hydroxyl values should be determinable using lifetime estimates and the quantities of other gases.
#

```{code-cell} ipython3
import pandas as pd
import scmdata
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
def explore_future_emissions(path,scenario,assumption,source):
    emissions= pd.read_csv(path)
    emissions.loc[emissions["region"]=="World"].groupby([
            "assumptions","model","modified","region","scenario","unit","variable"
            ]
        ).sum(numeric_only=True)
#     emissions = emissions.drop_meta(['sector','assumptions','modified','source'])
    out=scmdata.ScmRun(emissions).filter(source=source,sector='Total',region='World')
#     out=scmdata.ScmRun(emissions).filter(scenario=scenario,sector='Total',region='World')

    out=out.filter(variable = ["*|CH4","*|H2"]).drop_meta(['sector'])
    return out

path = 'datasets/emissions_total_scenarios.csv'
scenario = 'SSP2-45'
assumption='high'
source='adjusted'

future_hydrogen = explore_future_emissions(path,scenario,assumption,source)
future_hydrogen.timeseries()

for vdf in future_hydrogen.groupby('variable'):
    vdf.lineplot(hue='source',style='assumptions',size='scenario')
    plt.show()
```

```{code-cell} ipython3
for vdf1,vdf2 in zip(future_hydrogen.filter(variable="*|CH4").groupby('assumptions'),future_hydrogen.filter(variable="*|H2").groupby('assumptions')):
    vdf1.lineplot(style='scenario',color='b')
    ax2 = plt.twinx()
    vdf2.lineplot(style='scenario',ax=ax2)
    plt.ylim([0,80])
    plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
