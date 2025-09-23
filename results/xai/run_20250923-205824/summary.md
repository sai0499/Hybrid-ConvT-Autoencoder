# Explainability Report

Run directory: `results\xai\run_20250923-205824`

Method: attr_correlation
  Samples used: 4000 | latent_dim: 64 
  area_frac: z62 corr=0.267
  x_spread: z42 corr=-0.236
  y_spread: z31 corr=0.245
  edge_density: z10 corr=-0.170

Method: latent_embedding
  Samples used: 2000 | method: tsne 

Method: encoder_integrated_gradients
  Target dim: z0 | samples: 4 | baseline: white 

Method: encoder_gradcam
  Layer: encoder.res_skip | target dim: z0 | samples: 4

Method: encoder_gradcam
  Layer: encoder.res_skip | target dim: z17 | samples: 4

Method: decoder_influence
  Dimensions analysed: z0, z1, z2, z3

Method: counterfactual_nudge
  Dims traversed: [0, 1, 2] | delta: 1.0

Method: counterfactual_optimize
  Target area -> 0.35 | achieved 0.0313

Method: dice_counterfactuals
  Dims: [17, 19, 57, 29, 7, 10] | surrogate acc: 0.596 | samples: 4
