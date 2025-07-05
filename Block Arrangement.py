# Heuristic 1: Count of misplaced boxes
def h_misplaced_boxes(current, goal):
    count = 0
    for i in range(len(current)):
        if current[i] != goal[i]:
            count += 1
    return count

# Heuristic 2: Sum of distances boxes need to move (linear positions)
def h_sum_distances(current, goal):
    total = 0
    # Map box to its position in goal
    goal_positions = {box: idx for idx, box in enumerate(goal)}
    
    for idx, box in enumerate(current):
        goal_idx = goal_positions[box]
        total += abs(idx - goal_idx)
    return total

# Example usage
current_arrangement = ['B', 'A', 'D', 'C']
goal_arrangement = ['A', 'B', 'C', 'D']

print("Misplaced Boxes Heuristic:", h_misplaced_boxes(current_arrangement, goal_arrangement))
print("Sum of Distances Heuristic:", h_sum_distances(current_arrangement, goal_arrangement))
