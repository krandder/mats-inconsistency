"""Literature methods as aggbench baselines: official code where it exists, published equations otherwise.

Kelvin's rule: reimplement as little as possible; apply the published methods directly. Every method here records
where its code comes from in SOURCES. Modules:
  closed_form     PIF Gaussian oracle, Winkler GLS, covariance intersection, inverse covariance intersection,
                  extremizing (re-export), surprisingly popular (equations, cited per function)
  learned         Deep Sets and Set Transformer (authors' code, vendored) and TarMAC (MATE's reimplementation of
                  the attention step, vendored), with one thin token adapter
  tabpfn_adapter  TabPFN (PriorLabs pip package) as an in-context aggregator
  report          the default-config table in aggbench/README.md ("Literature methods")
torch and tabpfn are needed only by learned/tabpfn_adapter/report; closed_form needs numpy only.
"""

SOURCES = {
    "tabpfn": {"kind": "official package", "package": "tabpfn==9.2.0 (PyPI), weights TabPFN v2 (ModelVersion.V2)",
               "repo": "https://github.com/PriorLabs/TabPFN", "license": "Apache-2.0 (package); v2 weights under the "
               "Prior Labs licence (Apache-2.0 with attribution)", "paper": "Hollmann et al. 2025, Nature, "
               "doi:10.1038/s41586-024-08328-6; PFN: Mueller et al. 2022, arXiv:2112.10510"},
    "deepsets": {"kind": "official code (vendored)", "repo": "https://github.com/juho-lee/set_transformer",
                 "commit": "73432c640ac78140496d6738416c54d32c686d65", "file": "models.py: DeepSet",
                 "license": "MIT", "paper": "Zaheer et al. 2017, arXiv:1703.06114 (baseline as in Lee et al. 2019)"},
    "set_transformer": {"kind": "official code (vendored)", "repo": "https://github.com/juho-lee/set_transformer",
                        "commit": "73432c640ac78140496d6738416c54d32c686d65",
                        "file": "modules.py (MAB, SAB, ISAB, PMA), models.py: SetTransformer", "license": "MIT",
                        "paper": "Lee et al. 2019, arXiv:1810.00825"},
    "tarmac": {"kind": "third-party reimplementation (vendored) + paper equations",
               "repo": "https://github.com/XuehaiPan/mate", "commit": "3e631c0c3b043990fc53ae5fc3a37b0f65f230c5",
               "file": "examples/tarmac/models.py: MessageAggregator", "license": "MIT",
               "paper": "Das et al. 2019, arXiv:1810.11187 (no official code released; searched GitHub 2026-10-21)"},
    "pif_oracle": {"kind": "equations", "paper": "Satopaa, Jensen, Pemantle, Ungar, arXiv:1505.06472, Sec. 2.2.1 "
                   "eq. (1), Prop. 2.2 iii, Sec. 2.2.2 revealed aggregator"},
    "winkler_gls": {"kind": "equations", "paper": "Winkler 1981, doi:10.1287/mnsc.27.4.479; Clemen & Winkler 1985, "
                    "doi:10.1287/opre.33.2.427"},
    "covariance_intersection": {"kind": "equations", "paper": "Julier & Uhlmann 1997, doi:10.1109/ACC.1997.609105; "
                                "form of Noack et al. 2017 eq. (10a,b), N-source with det criterion"},
    "inverse_covariance_intersection": {"kind": "equations", "paper": "Noack, Sijs, Reinhardt, Hanebeck 2017, "
                                        "doi:10.1016/j.automatica.2017.01.019, Theorem 7, eqs. (17)-(18)"},
    "extremized_log_pool": {"kind": "equations (aggbench.baselines.extremized)",
                            "paper": "Satopaa et al. 2014, doi:10.1016/j.ijforecast.2013.09.009"},
    "surprisingly_popular": {"kind": "equations (no aggbench meta-predictions yet)",
                             "paper": "Prelec, Seung, McCoy 2017, doi:10.1038/nature21054"},
    "avici": {"kind": "not applied", "package": "avici 1.0.7 (PyPI), https://github.com/larslorch/avici, MIT",
              "paper": "Lorch et al. 2022, arXiv:2205.12934", "why": "see README"},
    "nebp": {"kind": "later item; no official code found", "paper": "Garcia Satorras & Welling 2021, "
             "arXiv:2003.01998", "related_official": "https://github.com/vgsatorras/hybrid-inference (arXiv:1906.02547)"},
}
