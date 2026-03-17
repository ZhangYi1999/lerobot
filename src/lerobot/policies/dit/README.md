```
velocity_net (_DiTNoiseNet)
├── time_net.out_net.{0,2}           — 2 Linear (time embedding)
├── ac_proj.{0,2}                     — 2 Linear (action projection)
├── cond_proj                         — 1 Linear (flat cond, use_encoder=False)
│   OR cond_proj.encoder.layers.N     — encoder layers (use_encoder=True)
│       ├── self_attn.out_proj        — Linear (only named submodule of MHA)
│       ├── mlp.linear1               — Linear
│       └── mlp.linear2               — Linear
├── decoder.layers.N (_DiTDecoder × 6)
│   ├── self_attn.out_proj            — Linear
│   ├── mlp.linear1                   — Linear
│   ├── mlp.linear2                   — Linear
│   ├── attn_modulate.{scale,shift}   — 2 Linear (AdaLN)
│   ├── attn_gate.scale               — Linear (zero-init gate)
│   ├── mlp_modulate.{scale,shift}    — 2 Linear (AdaLN)
│   └── mlp_gate.scale                — Linear (zero-init gate)
└── eps_out (_FinalLayer)
    ├── linear                        — Linear
    └── adaLN_modulation.1            — Linear
```