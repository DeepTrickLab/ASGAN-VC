import os

MODEL = [
    "AutoVC",
    "AutoVC2",
    "AutoVC3",
    "MetaVC",
    "MetaVC2",
    "MetaVC3",
]
PT_NAME = [
    "autovc_128_vctk",
    "autovc2_128_vctk",
    "autovc3_128_vctk",
    "autovc_gan_128_vctk",
    "autovc2_gan_128_vctk",
    "autovc3_gan_128_vctk",
    "autovc_sngan_128_vctk",
    "autovc2_sngan_128_vctk",
    "autovc3_sngan_128_vctk",
    "metavc_128_vctk",
    "metacv2_128_vctk",
    "metavc3_128_vctk",
    "metavc_gan_128_vctk",
    "metavc2_gan_128_vctk",
    "metavc3_gan_128_vctk",
    "metavc_sngan_128_vctk",
    "metavc2_sngan_128_vctk",
    "metavc3_sngan_128_vctk",
]

for pt_name in PT_NAME:
    tmp_name = pt_name[:7]
    if "m" in tmp_name:
        model_name = "MetaVC"
    else:
        model_name = "AutoVC"
    if "2" in tmp_name or "3" in tmp_name:
        is_adain = "True"
        if "2" in tmp_name:
            model_name += "2"
        else:
            model_name += "3"
        cmd = f"python evaluate_convert_with_thirdparty.py --model_name={model_name}  --pt_name={pt_name} --is_adain={is_adain}"
    else:
        cmd = f"python evaluate_convert_with_thirdparty.py --model_name={model_name}  --pt_name={pt_name}"
    os.system(cmd)
    os.system("cls||clear")
