# Step activation function
def step(x):
    return 1 if x >= 0 else 0  # Output is 1 if input is >= 0, else 0

# Training function for OR gate
def train_or_perceptron():
    # OR gate truth table: (x1, x2, target output)
    dataset = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1)
    ]
    
    # Initialize weights and bias
    w1 = 0.0
    w2 = 0.0
    bias = 0.0
    
    learning_rate = 0.1
    max_epochs = 100

    print("\nTraining Perceptron for OR Gate")
    print(f"Initial: w1 = {w1}, w2 = {w2}, bias = {bias}")
    
    for epoch in range(max_epochs):
        error_found = False  # To check if any update is needed
        
        print(f"\nEpoch {epoch + 1}")
        for x1, x2, target in dataset:
            # Calculate weighted sum + bias
            y_in = x1 * w1 + x2 * w2 + bias

            # Pass through step function to get output
            output = step(y_in)

            # Calculate error
            error = target - output

            # If there's an error, update weights and bias
            if error != 0:
                w1 += learning_rate * error * x1
                w2 += learning_rate * error * x2
                bias += learning_rate * error
                error_found = True  # Means we still need more training
            
            print(f"Input: ({x1}, {x2}) | Target: {target} | Output: {output} | "
                  f"w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {bias:.2f}")
        
        # If no error in the whole dataset, we can stop early
        if not error_found:
            print("\nTraining completed (converged).")
            break
    else:
        print("\nTraining stopped (max epochs reached).")

    # Final weights
    print(f"\nFinal Weights: w1 = {w1:.2f}, w2 = {w2:.2f}, bias = {bias:.2f}")
    
    # Final test
    print("\nTesting Trained Model:")
    for x1, x2, target in dataset:
        result = step(x1 * w1 + x2 * w2 + bias)
        print(f"Input: ({x1}, {x2}) | Predicted: {result} | Actual: {target}")

# Call the function to train OR gate
train_or_perceptron()
