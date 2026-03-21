import random

def get_decision(terrain):
    terrain = terrain.lower()

    if terrain == "smooth":
        return "Safe", random.randint(70, 100)
    elif terrain == "gravel":
        return "Moderate", random.randint(40, 60)
    elif terrain == "sand":
        return "High", random.randint(20, 40)
    elif terrain == "rock":
        return "Dangerous", random.randint(0, 10)
    else:
        return "Unknown", 0