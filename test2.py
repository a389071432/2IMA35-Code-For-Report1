import matplotlib.pyplot as plt
import numpy as np

ratio = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
success_rate = [0.4, 0.5, 0.5, 0.6, 0.6, 0.4, 0.45, 0.45, 0.6, 0.55, 0.75, 0.6, 0.8, 0.6, 0.75, 0.7]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(ratio, success_rate, 'b-o', linewidth=2, markersize=6)

# Add value labels above points
for x, y in zip(ratio, success_rate):
    plt.annotate(f'{y:.2f}', 
                (x, y), 
                textcoords="offset points", 
                xytext=(0,10),   # 10 points vertical offset
                ha='center')     # horizontal alignment

# Customize the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Size Ratio')
plt.ylabel('Success Rate')
plt.title('Success Rate vs Size Ratio (density fixed)')

# Set axis limits
plt.ylim(0, 1.1)  # Set y-axis from 0 to 1.1 to show full range and labels
plt.xlim(0.9, 2.6)  # Add some padding to x-axis

# Show the plot
plt.tight_layout()
plt.show()


def rate_of_change(arr):
   # Count changes between adjacent values
   changes = 0
   for i in range(1, len(arr)):
    changes += abs(arr[i] - arr[i-1])
   
   # Calculate rate as number of changes divided by potential changes
   rate = changes / (len(arr) - 1)
   return rate

# Example usage
success_rate = [0.4, 0.5, 0.5, 0.6, 0.6, 0.4, 0.45, 0.45, 0.6, 0.55, 0.75, 0.6, 0.8, 0.6, 0.75, 0.7]
roc = rate_of_change(success_rate)

print(f"Rate of change: {roc:.3f}")