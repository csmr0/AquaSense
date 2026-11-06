AquaSense: Milestone 3 — leak localization
=========================================

Requirements:
  Already working: python, wntr, numpy, pandas, scikit-learn, joblib
  network.inp (your leak-FREE EPANET export)
  Optional: ml_results/detector.joblib from Milestone 2

Files in your working folder:
  D:\VIT Hackathon\Prototype2\
    network.inp
    train_anomaly_detector.py
    localize_leak.py
    ml_results\detector.joblib

1. Generate a hidden leak automatically:
   python localize_leak.py
   python localize_leak.py --hidden-junction 5

2. Create a leak in EPANET yourself:
   - Open a COPY of the leak-free network.
   - Add an emitter coefficient such as 0.5 at one junction.
   - File > Export > Network as observed_leak.inp.
   - Keep network.inp leak-free.
   - Run: python localize_leak.py --observed-inp observed_leak.inp

   The detector reads only readings from the leak scenario, NOT the emitter's
   location. It makes its predictions against simulations from network.inp.

3. Additional simulation-only evaluation:
   python localize_leak.py --evaluate

The output ranks possible leak JUNCTIONS based on how closely EPANET-generated
pressure/flow signatures match the observed pressure/flow values. The ranking
is saved to ml_results/localization_ranking.csv. Scores are not probabilities.

Limitations:
- This model identifies candidate junctions only; do not claim pipe-level
  localization until you model internal pipe leaks via pipe-splitting.
- The model presumes a single leak and fits a few emitter sizes and global
  demand multipliers, which cannot represent all real operating conditions.
- Few sensors can make several leak locations indistinguishable.
- Estimated water lost assumes steady leak conditions, and source pipe "1"
  measures all network input flow.
- Evaluate on multiple hidden leak locations, changing demands, noise levels,
  and model parameter errors before making accuracy claims.
