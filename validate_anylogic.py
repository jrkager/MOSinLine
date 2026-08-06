import csv, sys, os, datetime

def run(folder, suffix, start_date="2025-03-03", warmup_days=14, weeks=52.0):
    p = lambda name: os.path.join(folder, f"{name}{suffix}.csv")
    t0 = datetime.date.fromisoformat(start_date)
    cut = t0 + datetime.timedelta(days=warmup_days)
    cutoff_h = warmup_days * 24.0

    demA = demB = demC = 0
    for row in csv.reader(open(p("planned_demands")), delimiter=";"):
        if len(row) < 5 or float(row[0]) < cutoff_h:
            continue
        demA += float(row[2]); demB += float(row[3]); demC += float(row[4])

    wA = wB = wC = 0; co2fw = 0.0
    r = csv.reader(open(p("WeeklyWasteReport")), delimiter=";"); next(r)
    for row in r:
        if not row or not row[0].isdigit() or int(row[0]) <= warmup_days // 7:
            continue
        wA += float(row[4]); wB += float(row[5]); wC += float(row[6]); co2fw += float(row[7])

    lost = 0; locost = 0.0
    r = csv.reader(open(p("WeeklyStockoutReport")), delimiter=";"); next(r)
    for row in r:
        if not row or not row[0].isdigit() or int(row[0]) <= warmup_days // 7:
            continue
        lost += float(row[3]); locost += float(row[7])

    km = fuel = co2g = 0.0; ntr = 0
    for row in csv.reader(open(p("trucks")), delimiter=";"):
        if len(row) < 8:
            continue
        try:
            d = datetime.datetime.strptime(row[1].split(",")[0].strip(), "%d/%m/%Y").date()
        except ValueError:
            continue
        if d < cut:
            continue
        km += float(row[2]); fuel += float(row[5]); co2g += float(row[6]); ntr += 1

    D = demA + demB + demC
    W = wA + wB + wC
    print(f"suffix {suffix}: demand {D} u ({D/weeks:.1f}/wk; A {demA} B {demB} C {demC})")
    print(f"{'run':<14}{'waste%':>8}{'so%':>7}{'FW_CO2':>8}{'TR_CO2':>8}{'TRcost':>8}{'km':>6}"
          f"{'routes':>7}{'so_cost':>8}")
    print(f"{'AnyLogic':<14}{100*W/D:>8.2f}{100*lost/D:>7.2f}{co2fw/weeks:>8.0f}"
          f"{co2g/1000/weeks:>8.0f}{fuel/weeks:>8.0f}{km/weeks:>6.0f}{ntr/weeks:>7.1f}"
          f"{locost/weeks:>8.0f}")
    print(f"(waste A/B/C: {wA}/{wB}/{wC}; lost total {lost})")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "D:/mosinline_anylogic_r101"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "_s1_variant2"
    run(folder, suffix)