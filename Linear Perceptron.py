# Linear activation function (no step function here, since we want continuous output)
def predict(x, w, b):
    return x * w + b  # This is our hypothesis: y = wx + b

# Train perceptron for fitting y = ax + b
def train_linear_perceptron():
    # Example training data: pairs of (x, y)
    # Let's say we are trying to learn y = 2x + 1
    dataset = [
        (0, 1),
        (1, 3),
        (2, 5),
        (3, 7),
        (4, 9)
    ]
    
    # Initialize weight (w) and bias (b)
    w = 0.0
    b = 0.0
    
    learning_rate = 0.01
    max_epochs = 100

    print("\nTraining Linear Perceptron for y = ax + b")
    print(f"Initial: weight = {w}, bias = {b}")
    
    for epoch in range(max_epochs):
        total_error = 0
        
        print(f"\nEpoch {epoch + 1}")
        for x, y_true in dataset:
            y_pred = predict(x, w, b)  # Predicted y
            
            error = y_true - y_pred  # Difference between actual and predicted
            total_error += error ** 2  # Accumulate squared error
            
            # Update rule for weight and bias using gradient descent
            w += learning_rate * error * x
            b += learning_rate * error
            
            print(f"Input: {x} | Target: {y_true} | Predicted: {y_pred:.2f} | "
                  f"Updated w = {w:.4f}, b = {b:.4f}")
        
        # Stop early if error is very small
        if total_error < 1e-5:
            print("\nTraining converged.")
            break
    else:
        print("\nTraining stopped (max epochs reached).")
    
    # Final model
    print(f"\nFinal Model: y = {w:.2f}x + {b:.2f}")
    
    # Test on training data
    print("\nTesting Trained Model:")
    for x, y_true in dataset:
        y_pred = predict(x, w, b)
        print(f"Input: {x} | Predicted: {y_pred:.2f} | Actual: {y_true}")

# Call the training function
train_linear_perceptron()
