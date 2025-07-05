from collections import Counter

# Manhattan distance for two numeric features
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# KNN prediction
def knn_predict(test_point, dataset, k=3):
    distances = []
    for (size, texture), label in dataset:
        dist = manhattan((size, texture), test_point)
        distances.append((dist, label))
    
    distances.sort()
    k_labels = [label for (_, label) in distances[:k]]
    return Counter(k_labels).most_common(1)[0][0]

# Example dataset: ((size, texture), category)
dataset = [
    ((4.0, 7.0), "A"),
    ((5.0, 6.0), "A"),
    ((6.0, 5.0), "B"),
    ((7.0, 4.0), "B"),
    ((8.0, 3.0), "C"),
    ((9.0, 2.0), "C"),
    ((3.0, 8.0), "D"),
    ((2.0, 9.0), "D")
]

# Test input: (size, texture)
test_point = (6.5, 4.5)

# Predict the category
k = 3
predicted_label = knn_predict(test_point, dataset, k)

print(f"\nTest Point: {test_point}")
print(f"Predicted Category (k={k}): {predicted_label}")
