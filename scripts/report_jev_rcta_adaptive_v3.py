"""Report v3 with the same six-case v2 comparison when those cases are present."""
from report_jev_rcta_adaptive_v2 import main

if __name__ == "__main__":
    main(protocol="jev-rcta-adaptive-v3", default_stem="jev_rcta_adaptive_v3_20260924",
         baseline_sets=(("JEV baseline", ("phase1",)),
                        ("Adaptive v2 point prediction", ("jev_rcta_adaptive_v2_smoke_20260924", "jev_rcta_adaptive_v2_long_20260924"))),
         test_report="reports/jev_rcta_adaptive_v3_tests_20260924.json")
