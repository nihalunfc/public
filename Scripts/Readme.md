# Dataset Description

## Overview
Simulated space-photometry light curves of 9,000 stars, each hosting exactly one transiting planet, with a noisy stellar catalog row per star and the full set of true simulation parameters. The data imitates what a wide-field space telescope records at 30-minute cadence over one to four 27-day observing segments: planetary transits, stellar spot modulation, flares and granulation noise, instrument systematics, data gaps and outliers. It is intended for research and benchmarking of methods that recover planet properties (orbital period, planet-to-star radius ratio) from raw light curves. Created in 2026 by the dataset author; no real mission data is used.

## How the data was collected
The data is synthetic and fully reproducible with the included script `make_dataset.py` (Python 3, numpy and pandas; pyarrow to write Parquet). Usage: `python make_dataset.py --seed SEED --out OUT_DIR`. The seed used for this release is not published.

Generation process, per star:
1. Star: spectral class (M 25%, K 30%, G 30%, F 15%), effective temperature, then mass and radius from main-sequence relations with 6% log-normal scatter; 12% of FGK stars are subgiants (1.5-3.2 times larger). Surface gravity follows from mass and radius. Apparent magnitude uniform from 8 to 15. Quadratic limb-darkening coefficients from temperature with scatter.
2. Planet: orbital period log-uniform 0.6-30 days (75%) or 30-300 days (25%); radius ratio log-uniform 0.012-0.25; impact parameter uniform up to 1 + radius_ratio/2 (grazing transits allowed); 30% eccentric orbits (Rayleigh 0.2, capped at 0.8); semi-major axis from Kepler's third law and the true stellar density.
3. Transit model: limb-darkened occultation integrated over 400 concentric stellar rings, Keplerian sky-projected separation, 7-point supersampling across each 30-minute exposure.
4. Stellar signals: rotation (period 0.5-80 days for M dwarfs, 2-40 days otherwise) with two harmonics whose amplitudes and phases evolve over 1-4 rotation periods (spot evolution); granulation as an Ornstein-Uhlenbeck process (20-300 ppm, 0.05-0.8 day timescale); flares with fast rise and exponential decay (frequent on M dwarfs).
5. Dilution: each campaign has a crowding level; stars in crowded campaigns are more likely to have unresolved neighbours (20-55% of stars), which contribute 2-70% of the measured flux. 20% of stars are also blended with a background eclipsing binary (period 0.4-25 days, separated from the planet period and its simple multiples; primary and secondary eclipses from a two-star limb-darkened eclipse model, often grazing and V-shaped) whose observed eclipses are 1.5-6 times deeper than the planet transit, contributing up to 30% of the flux. `crowding_obs` is a Poisson count of catalog sources in the aperture driven by the crowding level. The catalog contamination estimate is the true total contamination with 12% log-normal error plus a small positive offset, reduced by up to 60% for faint stars with many catalog neighbours (catalogs miss faint sources); missing for 8%.
6. Instrument: 48 observing campaigns, each with 1-4 segments of 27 days (back to back, or separated by 1, 2, 4, 8 or 12 segment lengths), a 1-day downlink gap mid-segment and random extra gaps. Per campaign: exponential ramps at the start of each orbit, scattered-light bumps near orbit ends, linear drifts, periodic attitude adjustments with flux jumps, outlier rate. Each star scales the campaign systematics with its own coefficients. Photon noise from magnitude (about 200 ppm per 30 minutes at magnitude 10, 40 ppm floor), 5% scatter in the reported error.
7. Outliers: positive and negative spikes; 70% are flagged in `quality`.
8. Normalisation: flux and flux_err are divided by the median of good points in each segment.
9. Acceptance: a draw is kept only if at least one transit falls in the observed data and the total transit signal-to-noise ratio, after all dilution, is at least 10.

## File Structure
- `stars.csv`: 9,000 rows, one per star: identifier, campaign, catalog columns and every true parameter.
- `lightcurves.parquet`: about 22.5 million rows, one per observation: star identifier, time, flux, flux error, quality flags.
- `campaigns.csv`: 48 rows: segment schedule and systematics settings of each campaign.
- `make_dataset.py`: the generator.

## Features
`stars.csv`:

| Column | Type | Description |
|--------|------|-------------|
| star_key | string | Star identifier |
| campaign | int | Observing campaign (0-47) |
| teff_obs, logg_obs, radius_obs, mass_obs | float | Catalog stellar parameters with noise (K, log10 cgs, solar radii, solar masses); about 5% missing each |
| magnitude | float | Apparent magnitude |
| contamination_obs | float | Catalog estimate of the flux fraction from neighbours; about 8% missing |
| crowding_obs | int | Catalog sources within the aperture |
| period | float | True orbital period, days |
| radius_ratio | float | True planet radius / stellar radius |
| impact, ecc, omega, a_rs, inc, t0 | float | True impact parameter, eccentricity, argument of periastron (rad), semi-major axis in stellar radii, inclination (rad), reference mid-transit time (days) |
| dilution | float | True flux fraction from static neighbours |
| total_dilution | float | True flux fraction from all neighbours including a blended binary |
| has_eb, eb_period, eb_frac, eb_depth_obs | int/float | Blended eclipsing binary present, its period (days), its flux fraction, its observed eclipse depth |
| crowding | float | True crowding level of the star's field |
| u1, u2 | float | True limb-darkening coefficients |
| teff, logg, radius, mass, spectral_class | float/string | True stellar parameters |
| prot, spot_amp | float | Rotation period (days) and spot-modulation amplitude |
| sigma | float | Photon noise per 30-minute point |
| snr, n_transits, n_points | float/int | Total transit signal-to-noise ratio, number of transits with observed data, number of observations |

`lightcurves.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| star_key | string | Star identifier |
| time | float | Days on the mission clock |
| flux | float | Normalised flux |
| flux_err | float | 1-sigma photon noise |
| quality | int | Bit flags: 1 attitude adjustment, 2 scattered light, 4 outlier |

## Notes
- Missing values: catalog columns only, as stated above; light-curve rows are complete (gaps are absent rows, not missing values).
- Edge cases included on purpose: blended eclipsing binaries, single-transit planets, transits falling in gaps, grazing transits, eccentric orbits, strongly spotted fast rotators, flare stars, faint stars near the detection limit, unflagged outliers.
- Licence: CC-BY-4.0.
