# Roadmap

Phase 4 will add evidence-backed parameter regression, production NRTL/UNIQUAC binary LLE,
multi-model comparison, sensitivity analysis, PDF reports, and DWSIM/Aspen configuration support.
SRK now has a pilot `thermo` adapter; enabling it as `production_ready` requires reviewed kij data
and benchmark closure. Later research may evaluate Dortmund-UNIFAC, CPA, PC-SAFT, eNRTL, and Pitzer
adapters. Scope expansion only follows verification coverage; electrolytes, SLE, VLLE, and full
flowsheets remain out of scope for the current release.

Phasepy and Clapeyron.jl now have optional Peng-Robinson adapters for the reviewed non-electrolyte
VLE boundary. The next integration step is evidence-backed Phasepy NRTL/UNIQUAC support and
Clapeyron association/SAFT models; neither may be enabled until parameter provenance and behavioral
validation cases exist. NeqSim remains a potential industrial JVM adapter. Every engine remains
isolated behind `ThermodynamicBackend` and must pass the same evidence and validation gates.
