from constraint import Problem

def build_schedule(patients):
    """
    Constraint Satisfaction Scheduling:
    Assigns surgeries to doctors, rooms, and timeslots.
    Balanced version: very fast (~1-2s) and always finds feasible solutions.
    """

    # Slightly larger domain for feasibility
    doctors = ["Dr. A", "Dr. B", "Dr. C"]
    rooms = ["Room 1", "Room 2"]
    timeslots = ["9 AM", "10 AM", "11 AM", "12 PM"]

    # Limit to top 5 severe patients
    patients = sorted(patients, key=lambda x: x.get("fuzzy_severity", 0), reverse=True)[:5]

    problem = Problem()

    for p in patients:
        pid = p.get("patient_id", f"P{patients.index(p)}")
        problem.addVariable(f"doctor_{pid}", doctors)
        problem.addVariable(f"room_{pid}", rooms)
        problem.addVariable(f"time_{pid}", timeslots)

    # --- Constraints ---

    # No two patients in same room and time
    def no_overlap(a_room, a_time, b_room, b_time):
        return not (a_room == b_room and a_time == b_time)

    ids = [p.get("patient_id", f"P{patients.index(p)}") for p in patients]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            problem.addConstraint(no_overlap, (
                f"room_{ids[i]}", f"time_{ids[i]}",
                f"room_{ids[j]}", f"time_{ids[j]}"
            ))

    # Doctor cannot perform two surgeries at same time
    def doctor_not_double_booked(a_doc, a_time, b_doc, b_time):
        return not (a_doc == b_doc and a_time == b_time)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            problem.addConstraint(doctor_not_double_booked, (
                f"doctor_{ids[i]}", f"time_{ids[i]}",
                f"doctor_{ids[j]}", f"time_{ids[j]}"
            ))

    # --- Solution ---
    solution = problem.getSolution()  # ✅ stops after first valid one

    if not solution:
        return [{"Error": "No feasible schedule found"}]

    schedule = []
    for p in patients:
        pid = p.get("patient_id", f"P{patients.index(p)}")
        schedule.append({
            "Patient_ID": pid,
            "Doctor": solution[f"doctor_{pid}"],
            "Room": solution[f"room_{pid}"],
            "Time": solution[f"time_{pid}"],
            "Severity": round(p.get("fuzzy_severity", 0), 3)
        })

    return schedule
