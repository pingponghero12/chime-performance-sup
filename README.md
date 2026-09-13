# CHIME parametric performance model

This repo hosts a lightweight, shareable model of the instrument performance of **CHIME**, the Copernicus
Hyperspectral Imaging Mission for the Environment. It is a reference implementation of the
SPIE paper *"CHIME System Performance at Critical Design Review: A parametric model for
end-to-end simulation analysis"* (Fernandez Nunez et al., ESA-ESTEC / OHB System AG /
ESA-ESRIN).

The model aims to support Sentinel User Preparation studies and similar initiatives that require knowledge of 
the CHIME performance beyond the system requirements and need to generate its own datasets

The model inputs oversampled TOA radiance and outpus CHIME like L1 data. As such it can be integrated in 
third party simulators that generate oversampled TOA radiance or be used within the proposed mini E2E reference 
implementation

It is **not** a replacement for dedicated E2E simulators like CHIME OPSI or CHEES
nor industrial verification tools, nor a rigorous per-pixel uncertainty propagation (CHIME ships that separately),
and not an orbit or direct-location framework. 


## Status

**Currently under development. Only parametric model has been committed

Still to come: A full mini E2E reference implementation including scene generation, atmospheric propagation and correction

Roadmap:
-Oct 2026 First formal issue including full E2E implementation
-Nov 2026 Iterations as part of SUP workshop in ESRIN
-Updates and maintenance throughout Phase D
-End of 2028 Update of Instrument modelling after Instrument level TVAC Characterisation 

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```


## Use
All data is stored in the folder data and loaded into an Instrument instance

Run example.ipynb for further details