import argparse, json, os, subprocess, sys

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(HERE, ".."))
EXE     = os.path.join(HERE, "gpu_opt_drc_methods")
CU      = os.path.join(HERE, "gpu_opt_drc_methods.cu")
TOPO    = os.path.join(HERE, "_topology.json")

def gds_to_json(gds_path):
    try:
        import pya
        layout = pya.Layout(); layout.read(gds_path)
        dbu = layout.dbu
        cell = layout.top_cells()[0]
        polys = []
        for li in layout.layer_indices():
            info = layout.get_info(li)
            it = cell.begin_shapes_rec(li)
            while not it.at_end():
                s = it.shape(); pts = None
                if s.is_polygon():
                    p = s.polygon.transformed(it.trans())
                    pts = [[round(pt.x*dbu,6), round(pt.y*dbu,6)]
                           for pt in p.each_point_hull()]
                elif s.is_box():
                    b = s.box.transformed(it.trans())
                    pts = [[round(b.left*dbu,6),  round(b.bottom*dbu,6)],
                           [round(b.right*dbu,6), round(b.bottom*dbu,6)],
                           [round(b.right*dbu,6), round(b.top*dbu,6)],
                           [round(b.left*dbu,6),  round(b.top*dbu,6)]]
                elif s.is_path():
                    p = s.path.polygon().transformed(it.trans())
                    pts = [[round(pt.x*dbu,6), round(pt.y*dbu,6)]
                           for pt in p.each_point_hull()]
                if pts and len(pts) >= 3:
                    polys.append({"layer": info.layer,
                                  "datatype": info.datatype,
                                  "points": pts})
                it.next()
        with open(TOPO, "w") as f: json.dump(polys, f)
        return len(polys)
    except ImportError:
        pass

    try:
        import gdstk
        lib  = gdstk.read_gds(gds_path)
        tops = lib.top_level()
        if not tops: return 0
        cell = tops[0]
        polys = []
        def add(p, layer, dt):
            pts = [[round(float(x),6), round(float(y),6)] for x,y in p.points]
            if len(pts) >= 3:
                polys.append({"layer": int(layer), "datatype": int(dt),
                              "points": pts})
        for p in cell.polygons: add(p, p.layer, p.datatype)
        for ref in cell.references:
            for p in ref.cell.polygons: add(p, p.layer, p.datatype)
        with open(TOPO, "w") as f: json.dump(polys, f)
        return len(polys)
    except ImportError:
        print("[ERROR] Install gdstk:  pip install gdstk")
        sys.exit(1)


def results_to_gds(viol_path, over_path, gds_path):
    import gdstk
    violations = json.load(open(viol_path)) if os.path.isfile(viol_path) else []
    overlaps   = json.load(open(over_path)) if os.path.isfile(over_path) else []
    lib  = gdstk.Library()
    cell = lib.new_cell("DRC_RESULTS")

    for v in violations:
        a, b = v["polyA"], v["polyB"]
        dist = v["distance"]
        cell.add(gdstk.rectangle((a["minX"],a["minY"]),(a["maxX"],a["maxY"]),layer=100))
        cell.add(gdstk.rectangle((b["minX"],b["minY"]),(b["maxX"],b["maxY"]),layer=101))
        mx = ((a["minX"]+a["maxX"]) + (b["minX"]+b["maxX"])) / 4
        my = ((a["minY"]+a["maxY"]) + (b["minY"]+b["maxY"])) / 4
        sz = max(dist * 0.4, 1.0)
        cell.add(gdstk.rectangle((mx-sz,my-sz),(mx+sz,my+sz),layer=102))

    for v in overlaps:
        o = v["overlap"]
        cell.add(gdstk.rectangle((o["minX"],o["minY"]),(o["maxX"],o["maxY"]),layer=200))

    lib.write_gds(gds_path)
    return len(violations), len(overlaps)


def build(arch):
    inc   = [f"-I{ROOT}", f"-I{os.path.join(ROOT,'external')}"]
    flags = ["-O2", f"-arch={arch}", "-std=c++17",
             "-Xcompiler", "/Zc:preprocessor,/EHsc"]
    r = subprocess.run(["nvcc", CU, "-o", EXE] + inc + flags,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] Build failed:\n{r.stderr}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gds",      default="input.gds")
    p.add_argument("--dist",     type=float, default=20.0)
    p.add_argument("--arch",     default="sm_75")
    p.add_argument("--no-build", action="store_true")
    args = p.parse_args()

    gds_path = args.gds if os.path.isabs(args.gds) \
               else os.path.join(HERE, args.gds)

    if not os.path.isfile(gds_path):
        print(f"[ERROR] GDS not found: {gds_path}")
        sys.exit(1)

    if not args.no_build:
        build(args.arch)

    n = gds_to_json(gds_path)
    if n == 0:
        print("[ERROR] No polygons found.")
        sys.exit(1)

    r = subprocess.run([EXE, TOPO, str(args.dist), HERE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] Engine failed:\n{r.stderr}")
        sys.exit(1)

    viol_path = os.path.join(HERE, "violations.json")
    over_path = os.path.join(HERE, "overlaps.json")
    rep_path  = os.path.join(HERE, "report.txt")
    out_gds   = os.path.join(HERE, "drc_results.gds")

    n_viol, n_over = results_to_gds(viol_path, over_path, out_gds)

    ms = "?"
    if os.path.isfile(rep_path):
        for line in open(rep_path):
            if "Time:" in line: ms = line.split(":")[1].strip()

    for f in [TOPO, viol_path, over_path, rep_path]:
        if os.path.isfile(f): os.remove(f)

    SEP = "=" * 50
    print(f"\n{SEP}")
    print(f"  DRC Results  —  {os.path.basename(gds_path)}")
    print(SEP)
    print(f"  Polygons:       {n}")
    print(f"  Min distance:   {args.dist} um")
    print(f"  Time:           {ms}")
    print()
    print(f"  Violations:     {n_viol}  (layer 100/101/102)")
    print(f"  Overlaps:       {n_over}  (layer 200)")
    print()
    print(f"  Output: drc_results.gds")
    print(f"  Layers in KLayout:")
    print(f"    100/101  polygon pair that violates distance rule")
    print(f"    102      gap marker")
    print(f"    200      critical overlap area")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
