import subprocess
import re
#  gcc -fopenmp -O3 -o matrix_multiply 1_2.c
# Number of iterations
num_iterations = 10
runtimes = []

# Regex pattern to extract the runtime from the output
runtime_pattern = r"Matrix multiplication completed in ([0-9.]+) seconds"

# Run the program 10 times
for _ in range(num_iterations):
    # Execute the C program and capture the output
    result = subprocess.run(["./matrix_multiply", "1.txt", "2.txt", "3.txt"],
                            capture_output=True, text=True)
    
    # Check if the output contains the runtime
    match = re.search(runtime_pattern, result.stdout)
    if match:
        runtime = float(match.group(1))
        runtimes.append(runtime)
    else:
        print("Failed to extract runtime from output:", result.stdout)
        continue

# Calculate average runtime and uncertainty
if runtimes:
    avg_runtime = sum(runtimes) / len(runtimes)
    uncertainty = (max(runtimes) - min(runtimes)) / 2

    # Display results
    print(f"Average Runtime: {avg_runtime:.6f} seconds")
    print(f"Uncertainty: ±{uncertainty:.6f} seconds")
else:
    print("No valid runtimes were extracted.")