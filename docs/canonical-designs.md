# Canonical panel designs

## Shared panel and intervention

The default configuration has $N_0=160$ never-treated units, $N_1=40$ treated
units, $T_0=40$ pre-treatment periods, and $T_1=10$ post-treatment periods.
Controls appear first and treated units appear last only in the default
canonical factory. The generic simulation infrastructure does not require this
order.

Treatment is absorbing and starts at zero-based period $T_0$. The realized
effect in the $k$th treated period is

$$
\tau_{it} = s k,
$$

with $s=0.2$ and $k=1,\ldots,T_1$. The default true ATT is therefore 1.1.
The observed outcome is

$$
Y_{it} = Y_{it}(0) + W_{it}\tau_{it}.
$$

All canonical designs add independent observation noise with variance 0.36.

## Factor path construction

A factor path is divided by its sample standard deviation with `ddof=1`. It is
not centered. The path types are:

- `drift`: a stationary AR(1) path with coefficient 0.5 plus $0.5t$;
- `cyclical`: $\sin(\pi t / 15)$;
- `trend`: $t$ plus standard normal noise;
- `ar1`: a stationary AR(1) path with coefficient 0.5;
- `white_noise`: independent standard normal values.

AR paths start from their stationary distribution. Drift and AR factor paths
discard 20 initial values before returning the requested horizon.

## Classic factor design

The untreated signal is

$$
Y_{it}(0) = \lambda_i^\top f_t + \varepsilon_{it},
$$

with two drift factors. Each control loading is drawn from
$N(-a,1)$ and each treated loading from $N(a,1)$. The `overlap` parameter is
$a$. The experiment uses $a=0$ for good loading overlap and $a=1$ for poor
overlap.

Factory: `classic_factor_design(overlap=...)`.

## Weak-factor design

This design uses five drift factors and five cyclical factors. Each standardized
factor is multiplied by 0.2. Loading distributions use the same overlap
parameter as the classic design.

Factory: `weak_factor_design(overlap=...)`.

## Sparse synthetic-control design

The control outcomes begin with a two-drift-factor signal. Candidate donor
scores are the sums of their two loadings. Units among the largest
`n_active` scores receive sampling weight 1; all other controls receive sampling
weight 0.1. The active donor set is sampled without replacement.

Active donors receive equal convex weights. Each treated signal is the active
donor average plus independent standard normal noise. The final latent signal
uses weight $1/(1+10^{-9})$ on the synthetic component and the remaining weight
on the original factor component.

The active count is

$$
n_{active} = \max\{1,\lceil \text{active_share} \times N_0 \rceil\}.
$$

Factory: `synthetic_control_design(active_share=...)`.

## Factor-synthetic mixture

This design uses one drift factor and one cyclical factor. The untreated latent
signal is an equal mixture of the factor signal and a synthetic-control signal.
The active donor count is $\min(N_0,N_1)$. Independent treated-component noise
has standard deviation 2 before the equal mixture is formed.

Factory: `factor_synthetic_design(overlap=...)`.

## Time-series design

Each unit has an independent process

$$
x_{it} = \phi x_{i,t-1} + \mu_i + \eta_{it},
$$

where $\eta_{it}$ is standard normal. Controls have $\mu_i=0$ and treated units
have $\mu_i=0.25$. The initial value is drawn from the stationary distribution,
so the stationary treated mean is $0.25/(1-\phi)$.

When `integrated=False`, $Y(0)$ uses $x_{it}$ plus observation noise. When
`integrated=True`, it uses the cumulative sum of $x_{it}$ plus observation
noise, which gives an ARIMA(1,1,0)-type path.

Factory: `time_series_design(coefficient=..., integrated=...)`.

## Mixed-factor design

The panel is divided into two equal-sized groups. The first group contains all
treated units, enough controls to fill half the panel, and two drift factors.
The second group contains only controls and two cyclical factors. Both
half-panels restart from the same random-generator state. The final unit order
returns all controls before treated units under the default assignment.

The total unit count must be even, and the number of treated units must be less
than half the panel.

Factory: `mixed_factor_design(overlap=...)`.

## Random-stream compatibility

The convenience functions preserve the NumPy draw order of the extracted
`panel_dgps` implementation. For a fixed NumPy version and seed, migrated draws
match that implementation cell for cell. The R reference uses a different
random-number generator, so cross-language comparisons must use distributional
properties or Monte Carlo uncertainty rather than identical seeded panels.

