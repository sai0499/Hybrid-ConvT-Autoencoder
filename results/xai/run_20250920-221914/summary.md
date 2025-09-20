# Explainability Report

Run directory: `results\xai\run_20250920-221914`

Method: attr_correlation
  Samples used: 4000 | latent_dim: 64 
  area_frac: z17 corr=-0.599
  x_spread: z33 corr=-0.245
  y_spread: z8 corr=0.282
  edge_density: z17 corr=-0.452

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
  Target area -> 0.35 | achieved 0.0323

Method: dice_counterfactuals
  Dims: [17, 19, 57, 29, 7, 10] | surrogate acc: 0.596 | samples: 4
