# Heuristic 1: Misplaced Tiles
def h_misplaced(current, goal):
    count = 0
    for i in range(3):
        for j in range(3):
            if current[i][j] != 0 and current[i][j] != goal[i][j]:
                count += 1
    return count

# Heuristic 2: Manhattan Distance
def h_manhattan(current, goal):
    total = 0
    # Create a dictionary to quickly find positions in the goal
    goal_pos = {}
    for i in range(3):
        for j in range(3):
            goal_pos[goal[i][j]] = (i, j)

    for i in range(3):
        for j in range(3):
            val = current[i][j]
            if val != 0:
                goal_i, goal_j = goal_pos[val]
                total += abs(i - goal_i) + abs(j - goal_j)
    return total


start_state = [
    [2, 8, 3],
    [1, 6, 4],
    [7, 0, 5]
]

goal_state = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]
]

print("Misplaced Tiles Heuristic:", h_misplaced(start_state, goal_state))
print("Manhattan Distance Heuristic:", h_manhattan(start_state, goal_state))
