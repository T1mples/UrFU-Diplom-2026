from algorithm import build_dihedral_involution_generators


THREE_INVOLUTIONS = "three_involutions"
ROTATION_REFLECTION = "rotation_reflection"
TWO_REFLECTIONS = "two_reflections"

ALL_FAMILIES = [
    THREE_INVOLUTIONS,
    ROTATION_REFLECTION,
    TWO_REFLECTIONS,
]

FAMILY_LABELS = {
    THREE_INVOLUTIONS: "три инволюции с коммутирующей парой",
    ROTATION_REFLECTION: "поворот, обратный поворот и отражение",
    TWO_REFLECTIONS: "две отражающие симметрии",
}


def iter_three_involution_systems(n):
    if n % 2 != 0:
        return

    for k in range(1, n):
        yield {
            "family": THREE_INVOLUTIONS,
            "parameters": f"k={k}",
            "k": k,
            "generators": build_dihedral_involution_generators(n, k),
        }


def iter_rotation_reflection_systems(n):
    for rotation_shift in range(1, n):
        inverse_shift = (-rotation_shift) % n
        for reflection_shift in range(n):
            yield {
                "family": ROTATION_REFLECTION,
                "parameters": f"a={rotation_shift}, b={reflection_shift}",
                "k": "",
                "generators": [
                    (rotation_shift, 0),
                    (inverse_shift, 0),
                    (reflection_shift, 1),
                ],
            }


def iter_two_reflection_systems(n):
    for first_shift in range(n):
        for second_shift in range(first_shift + 1, n):
            yield {
                "family": TWO_REFLECTIONS,
                "parameters": f"a={first_shift}, b={second_shift}",
                "k": "",
                "generators": [
                    (first_shift, 1),
                    (second_shift, 1),
                ],
            }


def iter_generator_systems(max_n, families=None):
    if families is None:
        families = [THREE_INVOLUTIONS]

    family_set = set(families)
    for n in range(2, max_n + 1, 2):
        if THREE_INVOLUTIONS in family_set:
            for system in iter_three_involution_systems(n):
                yield n, system
        if ROTATION_REFLECTION in family_set:
            for system in iter_rotation_reflection_systems(n):
                yield n, system
        if TWO_REFLECTIONS in family_set:
            for system in iter_two_reflection_systems(n):
                yield n, system
