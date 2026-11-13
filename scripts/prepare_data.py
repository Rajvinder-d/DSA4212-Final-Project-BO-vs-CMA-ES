from bo_vs_es.objectives.data import prepare

if __name__ == "__main__":
    raw, traces, ocv, manifest = prepare()
    for label, trace in traces.items():
        print(label, len(trace), "observations", "SOC", (trace.soc.min(), trace.soc.max()))
    print("Calibration-only OCV offset:", manifest["calibration_rest_offset_v"], "V")
