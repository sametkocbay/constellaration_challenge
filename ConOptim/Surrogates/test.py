import json
import jax.numpy as jnp
from constellaration import forward_model
from constellaration.geometry.surface_rz_fourier import SurfaceRZFourier
from constellaration.mhd.vmec_settings import VmecPresetSettings
import numpy as np

# The configuration string provided
stellarator_data = {
    "r_cos": [[0.0, 0.0, 0.0, 0.0, 1.0, 0.3178024006853376, -0.00494453968429039, -0.0010158379922691344,
               0.00014170255845398662],
              [0.000021273581625780166, -0.002504825752146429, -0.0019791928302356574, -0.05986009847084577,
               0.2378930884212573, 0.07041809177925817, -0.03405649158367229, -0.0012552908877077342,
               -0.000050578783214278185],
              [-0.000030068586955447527, -0.0008786797258682598, 0.00871051319329453, -0.006108510773329939,
               0.012799177446456245, 0.02540372085366101, 0.006120224656894394, 0.005782073039163714,
               -0.00032573388857629895],
              [-0.0003928164583617964, 0.0018402574023466017, -0.0028694583032823347, 0.003249729685005616,
               -0.00340277054620028, 0.0028927525886110798, -0.0058754202804079175, 0.0009349924265612791,
               -0.00029069423934959806],
              [-0.00013796254756594703, -0.00007113785778163367, -0.00007005393032934321, -0.00012567305130725142,
               -0.0000755348144625171, -0.00005890789481852012, -0.00016787611941031008, -0.0001681718530434264,
               -0.0007577573777222754]],
    "z_sin": [[0.0, 0.0, 0.0, 0.0, 0.0, -0.3682437645980358, -0.010313325093545838, 0.0008028627195158811,
               0.00008731728723274532],
              [-0.00021820594976462235, 0.00257055829045463, -0.0127795602890544, -0.05705253192342194,
               0.25012256718258646, 0.012207198333313168, 0.0340313223723876, 0.0003576776007283744,
               -0.0002845557347781906],
              [0.0006250656697134858, 0.0001351500860080824, 0.02077775360036836, -0.009487259768838647,
               0.023875626799357026, 0.017665643652408247, 0.03202330538363405, -0.002402806268419791,
               -0.00024556614589377266],
              [0.0003909828647171795, -0.0009267683930298048, 0.002547132301658598, 0.0018252150534381778,
               0.0025447442599817994, -0.0006139539204201416, 0.0040519500435168615, -0.002119370745245054,
               0.0006644491009208615],
              [0.0003314760556756545, 0.0003278655337955934, 0.00013156289898347628, -0.000025890394467765182,
               0.0005822073549505646, 0.000030278685234777375, -0.0001386996202989576, 0.0005453186709654603,
               0.00024046539821892854]],
    "n_field_periods": 3
}


def run_vmec_analysis():
    print("Reconstructing Surface...")
    # FIX: Use np.array instead of jnp.array
    boundary = SurfaceRZFourier(
        r_cos=np.array(stellarator_data["r_cos"], dtype=np.float64),
        z_sin=np.array(stellarator_data["z_sin"], dtype=np.float64),
        n_field_periods=stellarator_data["n_field_periods"],
        is_stellarator_symmetric=True
    )

    settings = forward_model.ConstellarationSettings(
        vmec_preset_settings=VmecPresetSettings(fidelity="low_fidelity"),
        qi_settings=None,
        turbulent_settings=None
    )

    print("Running VMEC Forward Model...")
    try:
        metrics, _ = forward_model.forward_model(boundary, settings=settings)

        print("\n" + "=" * 40)
        print("      VMEC GROUND TRUTH RESULTS")
        print("=" * 40)
        print(f"Max Elongation:     {metrics.max_elongation:.6f}")
        print(f"Aspect Ratio:       {metrics.aspect_ratio:.6f}")
        print(f"Avg Triangularity:  {metrics.average_triangularity:.6f}")
        print(f"Rot. Transform (ι): {metrics.edge_rotational_transform_over_n_field_periods:.6f}")
        print("=" * 40)

    except Exception as e:
        print(f"VMEC failed to converge: {e}")


if __name__ == "__main__":
    run_vmec_analysis()