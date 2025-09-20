# Explainability Report

Run directory: `results\xai\run_20250920-194223`

Method: attr_correlation
  Samples used: 400 | latent_dim: 64 
  area_frac: z17 corr=-0.588
  x_spread: z56 corr=0.275
  y_spread: z13 corr=-0.299
  edge_density: z17 corr=-0.456

Method: latent_embedding
  Samples used: 200 | method: tsne 

Method: encoder_integrated_gradients
  Target dim: z0 | samples: 4 | baseline: white 

Method: decoder_influence
  Dimensions analysed: z0, z1

Method: counterfactual_nudge
  Dims traversed: [0, 1] | delta: 1.0

Method: counterfactual_optimize
  Target area -> 0.4 | achieved 0.0323
