import heapq
import pandas as pd

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal, grid_size):
    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return g_score[current]

        x, y = current
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return float("inf")

def allocate_beds(patients, grid_size=6):
    beds = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    start_pos = (0, 0)
    patients_sorted = sorted(patients, key=lambda x: x.get("fuzzy_severity", 0), reverse=True)

    allocations = []
    used_beds = set()

    for p in patients_sorted:
        best_bed, best_cost = None, float("inf")
        for bed in beds:
            if bed in used_beds:
                continue
            cost = a_star_search(start_pos, bed, grid_size)
            if cost < best_cost:
                best_bed, best_cost = bed, cost

        if best_bed is None:
            for bed in beds:
                if bed not in used_beds:
                    best_bed = bed
                    best_cost = None
                    break

        used_beds.add(best_bed)
        allocations.append({
            "Patient": p.get("patient_id", "Unknown"),
            "Severity": round(p.get("fuzzy_severity", 0), 3),
            "Assigned_Bed": f"Bed-{best_bed[0]}-{best_bed[1]}" if best_bed else "None",
            "Distance_Cost": best_cost
        })

    df_alloc = pd.DataFrame(allocations)
    if "Distance_Cost" in df_alloc.columns:
        df_alloc["Distance_Cost"] = pd.to_numeric(df_alloc["Distance_Cost"], errors="coerce")

    return df_alloc.to_dict(orient="records")
