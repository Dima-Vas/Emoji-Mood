import matplotlib.pyplot as plt

# Read data from file
with open('output_smile.txt', 'r') as file:
    data = [float(line.strip()) for line in file]

# Plot the data
plt.plot(data)
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.title('Loss over epochs')
plt.grid(True)
plt.show()
