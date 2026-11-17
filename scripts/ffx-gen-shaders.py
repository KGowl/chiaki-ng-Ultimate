#!/usr/bin/env python3
"""Generates the Vulkan shader permutation headers of the AMD FidelityFX SDK 1.1.4
optical flow and frame interpolation components for chiaki-ng's Linux build.

The SDK produces them with FidelityFX_SC.exe (Windows only). This script does the
same job for the GLSL path with glslangValidator:

  * every combination of the permutation options is compiled (FFX_HALF=0/1 gives
    the plain and "_16bit" variants; "_wave64" variants are the same SPIR-V, GLSL
    has no way to request a wave size),
  * identical SPIR-V is stored once (named by its MD5),
  * the descriptor bindings are read from the SPIR-V itself (small parser below)
    and classified like FidelityFX_SC does (uniform buffer -> CBV, sampled image
    -> TextureSRV, storage image -> TextureUAV, read-only storage buffer ->
    BufferSRV, other storage buffer -> BufferUAV, sampler -> Sampler),
  * <name>_<md5>.h (reflection arrays + SPIR-V) and <name>_permutations.h
    (PermutationKey union, IndirectionTable, PermutationInfo table) are written in
    the layout the SDK's blob accessors include.

usage: ffx-gen-shaders.py <sdk dir> <out dir> [glslangValidator]
The output is committed (third-party/ffx-sdk/generated); rerun after updating the SDK.
"""
import concurrent.futures
import hashlib
import itertools
import os
import struct
import subprocess
import sys
import tempfile

# --- SPIR-V reflection -------------------------------------------------------

OP_NAME = 5
OP_TYPE_IMAGE = 25
OP_TYPE_SAMPLER = 26
OP_TYPE_SAMPLED_IMAGE = 27
OP_TYPE_ARRAY = 28
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_CONSTANT = 43
OP_VARIABLE = 59
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72

DEC_BLOCK = 2
DEC_BUFFER_BLOCK = 3
DEC_NON_WRITABLE = 24
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34

SC_UNIFORM_CONSTANT = 0
SC_UNIFORM = 2
SC_STORAGE_BUFFER = 12


def reflect(spirv):
    """Returns [(kind, name, binding, count, set)] sorted by (set, binding)."""
    words = struct.unpack("<%dI" % (len(spirv) // 4), spirv)
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    names, decos, member_nonwritable, member_count = {}, {}, {}, {}
    types, constants, variables = {}, {}, []
    i = 5
    while i < len(words):
        op = words[i] & 0xFFFF
        n = words[i] >> 16
        ins = words[i + 1:i + n]
        if op == OP_NAME:
            raw = struct.pack("<%dI" % (len(ins) - 1), *ins[1:])
            names[ins[0]] = raw.split(b"\0", 1)[0].decode()
        elif op == OP_DECORATE:
            decos.setdefault(ins[0], {})[ins[1]] = ins[2] if len(ins) > 2 else True
        elif op == OP_MEMBER_DECORATE:
            if ins[2] == DEC_NON_WRITABLE:
                member_nonwritable.setdefault(ins[0], set()).add(ins[1])
        elif op == OP_TYPE_IMAGE:
            types[ins[0]] = ("image", ins[1], ins[2], ins[6])  # sampled_type, dim, sampled (1 = SRV, 2 = UAV)
        elif op == OP_TYPE_SAMPLER:
            types[ins[0]] = ("sampler",)
        elif op == OP_TYPE_SAMPLED_IMAGE:
            types[ins[0]] = ("sampled_image", ins[1])
        elif op == OP_TYPE_ARRAY:
            types[ins[0]] = ("array", ins[1], ins[2])
        elif op == OP_TYPE_RUNTIME_ARRAY:
            types[ins[0]] = ("runtime_array", ins[1])
        elif op == OP_TYPE_STRUCT:
            types[ins[0]] = ("struct", list(ins[1:]))
            member_count[ins[0]] = len(ins) - 1
        elif op == OP_TYPE_POINTER:
            types[ins[0]] = ("pointer", ins[1], ins[2])
        elif op == OP_CONSTANT:
            constants[ins[1]] = ins[2]
        elif op == OP_VARIABLE:
            variables.append((ins[0], ins[1], ins[2]))
        i += n

    out = []
    for type_id, var_id, storage in variables:
        if storage not in (SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_STORAGE_BUFFER):
            continue
        d = decos.get(var_id, {})
        if DEC_BINDING not in d:
            continue
        tid = types[type_id][2]  # pointee
        count = 1
        while types[tid][0] in ("array", "runtime_array"):
            count = constants.get(types[tid][2], 1) * count if types[tid][0] == "array" else 0
            tid = types[tid][1]
        t = types[tid]
        if t[0] == "sampler":
            kind = "Sampler"
        elif t[0] == "sampled_image":
            kind = "TextureSRV"
        elif t[0] == "image":
            kind = "TextureUAV" if t[3] == 2 else "TextureSRV"
        elif t[0] == "struct":
            struct_id = tid
            sd = decos.get(struct_id, {})
            if storage == SC_UNIFORM and DEC_BLOCK in sd:
                kind = "CBV"
            else:
                readonly = d.get(DEC_NON_WRITABLE) or (
                    member_count.get(struct_id, 0) > 0 and
                    len(member_nonwritable.get(struct_id, ())) == member_count[struct_id])
                kind = "BufferSRV" if readonly else "BufferUAV"
            if not names.get(var_id):
                names[var_id] = names.get(struct_id, "")
        else:
            continue
        out.append((kind, names.get(var_id, ""), d[DEC_BINDING], count, d.get(DEC_DESCRIPTOR_SET, 0)))
    return sorted(out, key=lambda r: (r[4], r[2]))


# --- permutations ------------------------------------------------------------

KINDS = ["CBV", "TextureSRV", "TextureUAV", "BufferSRV", "BufferUAV", "Sampler", "RTAccelerationStructure"]
MEMBERS = [("ConstantBuffers", "constantBuffer"), ("SRVTextures", "srvTexture"), ("UAVTextures", "uavTexture"),
           ("SRVBuffers", "srvBuffer"), ("UAVBuffers", "uavBuffer"), ("Samplers", "sampler"),
           ("RTAccelerationStructures", "rtAccelerationStructure")]


def compile_one(glslang, src, includes, defines, tmp):
    fd, spv = tempfile.mkstemp(suffix=".spv", dir=tmp)
    os.close(fd)
    cmd = [glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-Os"] + \
          ["-D" + d for d in defines] + ["-I" + inc for inc in includes] + ["-o", spv, src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s %s failed:\n%s%s" % (os.path.basename(src), " ".join(defines), r.stdout, r.stderr))
    with open(spv, "rb") as f:
        blob = f.read()
    os.remove(spv)
    return blob


def generate(pool, glslang, out, name, src, includes, fixed, options, tmp):
    """options: [(OPTION_NAME, [values])]; each option gets enough key bits."""
    bits = [max(1, (len(v) - 1).bit_length()) for _, v in options]
    combos = list(itertools.product(*[range(len(v)) for _, v in options])) if options else [()]
    jobs = []
    for combo in combos:
        key, shift, defines = 0, 0, list(fixed)
        for (oname, vals), vi, b in zip(options, combo, bits):
            defines.append("%s=%s" % (oname, vals[vi]))
            key |= vi << shift
            shift += b
        jobs.append((key, pool.submit(compile_one, glslang, src, includes, defines, tmp)))
    uniques, index_of, table = [], {}, {}
    for key, fut in jobs:
        blob = fut.result()
        digest = hashlib.md5(blob).hexdigest()
        if digest not in index_of:
            index_of[digest] = len(uniques)
            uniques.append((digest, blob, reflect(blob)))
        table[key] = index_of[digest]

    for digest, blob, refl in uniques:
        pname = "%s_%s" % (name, digest)
        with open(os.path.join(out, pname + ".h"), "w") as fp:
            fp.write("// %s.h, generated by chiaki-ng scripts/ffx-gen-shaders.py\n\n" % pname)
            for k in KINDS:
                rs = [r for r in refl if r[0] == k]
                if not rs:
                    continue
                fp.write("static const char* g_%s_%sResourceNames[] = { %s };\n" % (pname, k, " ".join('"%s",' % r[1] for r in rs)))
                fp.write("static const uint32_t g_%s_%sResourceBindings[] = { %s };\n" % (pname, k, " ".join("%d," % r[2] for r in rs)))
                fp.write("static const uint32_t g_%s_%sResourceCounts[] = { %s };\n" % (pname, k, " ".join("%d," % r[3] for r in rs)))
                fp.write("static const uint32_t g_%s_%sResourceSets[] = { %s };\n\n" % (pname, k, " ".join("%d," % r[4] for r in rs)))
            fp.write("static const uint32_t g_%s_size = %d;\n\n" % (pname, len(blob)))
            fp.write("static const unsigned char g_%s_data[] = {\n" % pname)
            for j in range(0, len(blob), 32):
                fp.write(",".join("0x%02x" % b for b in blob[j:j + 32]) + ",\n")
            fp.write("};\n\n")

    with open(os.path.join(out, "%s_permutations.h" % name), "w") as fp:
        fp.write("// %s_permutations.h, generated by chiaki-ng scripts/ffx-gen-shaders.py\n\n" % name)
        for digest, _, _ in uniques:
            fp.write('#include "%s_%s.h"\n' % (name, digest))
        fp.write("\ntypedef union %s_PermutationKey {\n    struct {\n" % name)
        for (oname, _), b in zip(options, bits):
            fp.write("        uint32_t %s : %d;\n" % (oname, b))
        if not options:
            fp.write("        uint32_t unused : 1;\n")
        fp.write("    };\n    uint32_t index;\n} %s_PermutationKey;\n\n" % name)
        fp.write("typedef struct %s_PermutationInfo {\n    const uint32_t       blobSize;\n    const unsigned char* blobData;\n" % name)
        for plural, single in MEMBERS:
            fp.write("\n    const uint32_t  num%s;\n    const char**    %sNames;\n    const uint32_t* %sBindings;\n"
                     "    const uint32_t* %sCounts;\n    const uint32_t* %sSpaces;\n" % (plural, single, single, single, single))
        fp.write("} %s_PermutationInfo;\n\n" % name)
        total = 1 << sum(bits) if options else 1
        fp.write("static const uint32_t g_%s_IndirectionTable[] = {\n" % name)
        for k in range(total):
            fp.write("    %d,\n" % table.get(k, 0))
        fp.write("};\n\n")
        fp.write("static const %s_PermutationInfo g_%s_PermutationInfo[] = {\n" % (name, name))
        for digest, blob, refl in uniques:
            pname = "%s_%s" % (name, digest)
            fp.write("    { g_%s_size, g_%s_data, " % (pname, pname))
            for k in KINDS:
                n = sum(1 for r in refl if r[0] == k)
                if n:
                    fp.write("%d, g_%s_%sResourceNames, g_%s_%sResourceBindings, g_%s_%sResourceCounts, g_%s_%sResourceSets, "
                             % (n, pname, k, pname, k, pname, k, pname, k))
                else:
                    fp.write("0, 0, 0, 0, 0, ")
            fp.write("},\n")
        fp.write("};\n\n")
    return len(combos), len(uniques)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sdk, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    glslang = sys.argv[3] if len(sys.argv) > 3 else "glslangValidator"
    gpu = os.path.join(sdk, "include", "FidelityFX", "gpu")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".h"):
            os.remove(os.path.join(out, f))

    fi_fixed = ["FFX_GLSL=1", "FFX_GPU=1",
                "FFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
                "FFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
                "FFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
                "FFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
                "FFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"]
    fi_options = [("FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS", ["0", "1"]),
                  ("FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS", ["0", "1"]),
                  ("FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH", ["0", "1"])]
    of_fixed = ["FFX_GLSL=1", "FFX_GPU=1"]
    of_options = [("FFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT", ["0", "1"])]

    work = []
    fi_dir = os.path.join(sdk, "src", "backends", "vk", "shaders", "frameinterpolation")
    for f in sorted(os.listdir(fi_dir)):
        if f.endswith(".glsl"):
            work.append(("frameinterpolation", os.path.join(fi_dir, f), fi_fixed, fi_options))
    of_dir = os.path.join(sdk, "src", "backends", "vk", "shaders", "opticalflow")
    for f in sorted(os.listdir(of_dir)):
        if f.endswith(".glsl"):
            work.append(("opticalflow", os.path.join(of_dir, f), of_fixed, of_options))

    tmp = tempfile.mkdtemp(prefix="ffxgen_")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for comp, src, fixed, options in work:
            stem = os.path.splitext(os.path.basename(src))[0]
            includes = [gpu, os.path.join(gpu, comp)]
            for variant, half in (("", 0), ("_wave64", 0), ("_16bit", 1), ("_wave64_16bit", 1)):
                combos, uniques = generate(pool, glslang, out, stem + variant, src, includes,
                                           fixed + ["FFX_HALF=%d" % half], options, tmp)
                print("%s%s: %d permutations, %d unique" % (stem, variant, combos, uniques))
    os.rmdir(tmp)


if __name__ == "__main__":
    main()
