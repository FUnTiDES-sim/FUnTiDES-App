"""
Example: create synthetic SEP models and read them back.

Two models are built over a 1000 x 1000 x 1000 m domain at 10 m resolution:

  1. Layered model  -- three horizontal layers with increasing Vp.
  2. Salt model     -- same layering with an ellipsoidal high-velocity inclusion.

The script then reads each model back with :func:`~makutu.input.SEPReader.read_sep`
and prints basic statistics so you can verify the values are correct.

Usage
-----
    python heterogeneous_model_example.py [--output-dir DIR]

Output files
------------
    <output-dir>/vp_layered.H        Vp layered model header
    <output-dir>/vp_layered.H.@      Vp layered model binary
    <output-dir>/vp_layered_rho.H    Rho layered model header
    <output-dir>/vp_layered_rho.H.@  Rho layered model binary
    <output-dir>/vp_salt.H           Vp salt model header
    <output-dir>/vp_salt.H.@         Vp salt model binary
    <output-dir>/vp_salt_rho.H       Rho salt model header
    <output-dir>/vp_salt_rho.H.@     Rho salt model binary
"""

import argparse
import os

import numpy as np

from makutu.input.createSepModel import inclusion_model, layered_model
from makutu.input.SEPReader import read_sep


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
LX, LY, LZ = 1000.0, 1000.0, 1000.0   # domain extents [m]
NX, NY, NZ  = 100,   100,    100       # grid samples  (10 m resolution)

# ---------------------------------------------------------------------------
# Background layering
# ---------------------------------------------------------------------------
#   0 – 300 m   : water / shallow sediments   Vp = 1500 m/s
#   300 – 600 m : compacted sediments         Vp = 2500 m/s
#   600 – 1000 m: deep sediments / basement   Vp = 3500 m/s
LAYERS = [
    {"z_top":   0.0, "vp": 1500.0, "rho": 1.0},
    {"z_top": 300.0, "vp": 2500.0, "rho": 1.7},
    {"z_top": 600.0, "vp": 3500.0, "rho": 2.2},
]

# ---------------------------------------------------------------------------
# Salt inclusion  (centred at 500/500/550 m, semi-axes 180/140/100 m)
# ---------------------------------------------------------------------------
SALT = {
    "cx": 500.0, "cy": 500.0, "cz": 550.0,
    "rx": 180.0, "ry": 140.0, "rz": 100.0,
    "vp": 4480.0,   # typical halite P-wave velocity
    "rho": 2.2,
}


def print_model_stats(label, sep_model):
    """Print min/max/mean of a SEPModel."""
    d = sep_model.data
    print(f"  {label}")
    print(f"    shape : ({sep_model.n3}, {sep_model.n2}, {sep_model.n1})"
          f"  [nz, ny, nx]")
    print(f"    grid  : dx={sep_model.d1:.1f} m  dy={sep_model.d2:.1f} m"
          f"  dz={sep_model.d3:.1f} m")
    print(f"    range : min={d.min():.1f}  max={d.max():.1f}"
          f"  mean={d.mean():.1f}  dtype={d.dtype}")
    unique_vals = np.unique(d)
    if len(unique_vals) <= 10:
        print(f"    values: {unique_vals.tolist()}")
    else:
        print(f"    unique values: {len(unique_vals)}"
              f"  (10th pctile={np.percentile(d,10):.1f}"
              f"  median={np.median(d):.1f}"
              f"  90th pctile={np.percentile(d,90):.1f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="./sep_models",
                        help="Directory for output SEP files (default: ./sep_models)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Build layered model
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Creating layered model ...")
    vp_path_lay = os.path.join(args.output_dir, "vp_layered.H")
    print(f"  Building layered model with {len(LAYERS)} layers:")
    print(f"  vp_path_lay: {vp_path_lay}")
    for i, layer in enumerate(LAYERS):
        print(f"    Layer {i}: z_top={layer['z_top']}, vp={layer['vp']}, rho={layer['rho']}")
    vp_lay, rho_lay = layered_model(
        vp_path_lay,
        LX, LY, LZ, NX, NY, NZ,
        LAYERS,
    )
    print(f"  Written: {vp_path_lay}")
    print(f"  Written: {vp_path_lay[:-2]}_rho.H")

    # -----------------------------------------------------------------------
    # 2. Build salt model
    # -----------------------------------------------------------------------
    print("\nCreating salt-body model ...")
    vp_path_salt = os.path.join(args.output_dir, "vp_salt.H")
    vp_salt, rho_salt = inclusion_model(
        vp_path_salt,
        LX, LY, LZ, NX, NY, NZ,
        LAYERS,
        SALT,
    )
    print(f"  Written: {vp_path_salt}")
    print(f"  Written: {vp_path_salt[:-2]}_rho.H")

    # -----------------------------------------------------------------------
    # 3. Read back and print statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Reading models back ...")

    vp_lay_read   = read_sep(vp_path_lay)
    rho_lay_read  = read_sep(os.path.join(args.output_dir, "vp_layered_rho.H"))
    vp_salt_read  = read_sep(vp_path_salt)
    rho_salt_read = read_sep(os.path.join(args.output_dir, "vp_salt_rho.H"))

    print("\nLayered model:")
    print_model_stats("Vp  [m/s]", vp_lay_read)
    print_model_stats("Rho [a.u.]", rho_lay_read)

    print("\nSalt model:")
    print_model_stats("Vp  [m/s]", vp_salt_read)
    print_model_stats("Rho [a.u.]", rho_salt_read)

    # -----------------------------------------------------------------------
    # 4. Verify roundtrip: values written == values read back
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Roundtrip verification ...")
    assert np.allclose(vp_lay.data,  vp_lay_read.data),  "Vp layered mismatch"
    assert np.allclose(rho_lay.data, rho_lay_read.data), "Rho layered mismatch"
    assert np.allclose(vp_salt.data,  vp_salt_read.data),  "Vp salt mismatch"
    assert np.allclose(rho_salt.data, rho_salt_read.data), "Rho salt mismatch"
    print("  All roundtrip checks passed.")

    # -----------------------------------------------------------------------
    # 5. Show a vertical Vp profile at the domain centre
    # -----------------------------------------------------------------------
    print("\nVertical Vp profile at (x=500m, y=500m):")
    ix = NX // 2
    iy = NY // 2
    z_coords = (np.arange(NZ) + 0.5) * (LZ / NZ)
    vp_profile = vp_salt_read.data[:, iy, ix]   # shape (NZ,)
    prev_vp = None
    for iz, (z, vp_val) in enumerate(zip(z_coords, vp_profile)):
        if vp_val != prev_vp:
            print(f"    z={z:6.1f} m  Vp={vp_val:.0f} m/s")
            prev_vp = vp_val


if __name__ == "__main__":
    main()
