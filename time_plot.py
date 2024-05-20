import pandas as pd
import matplotlib.pyplot as plt

seq_len = (64,100,128,256,512)#,1024)
embed_dim = (32,64,128,256,512)#,1024)

# Load the CSV file
data = pd.read_csv('test_times_2heads_PostNorm.csv')

# Extract the columns
x = data.iloc[:, 0]
y = data.iloc[:, 1]
z1 = data.iloc[:, 2]
z2 = data.iloc[:, 3]

maxVal = max(z1.max(), z2.max())
minVal = min(z1.min(), z2.min())
colorVal = minVal + 0.8 * (maxVal - minVal)

# Create the heat maps
#plt.subplot(1, 2, 1)
plt.imshow(z1.values.reshape((len(set(y)), len(set(x)))), cmap='hot', origin='lower', vmin=minVal, vmax=maxVal)
plt.colorbar()
plt.xlabel('Sequence Length')
plt.ylabel('Embedding Dimension')
plt.xticks(range(len(set(x))), seq_len)
plt.yticks(range(len(set(y))), embed_dim)
plt.title('Fourier Block Time (s)')

print(len(x), len(y))

# Add value labels to the heat map
for i in range(len(seq_len)):
    for j in range(len(embed_dim)):
        plt.text(j, i, f"{z1.values[i * len(embed_dim) + j]:.4f}", ha='center', va='center', color='black' if z1.values[i * len(embed_dim) + j] > colorVal else "white")

plt.tight_layout()
plt.savefig("FourierTime_2Heads_PostNorm.png", bbox_inches='tight', dpi=400)
plt.clf()

#plt.subplot(1, 2, 2)
plt.imshow(z2.values.reshape((len(set(y)), len(set(x)))), cmap='hot', origin='lower', vmin=minVal, vmax=maxVal)
plt.colorbar()
plt.xlabel('Sequence Length')
plt.ylabel('Embedding Dimension')
plt.xticks(range(len(set(x))), seq_len)
plt.yticks(range(len(set(y))), embed_dim)
plt.title('Attention Block Time (s)')

# Add value labels to the heat map
for i in range(len(seq_len)):
    for j in range(len(embed_dim)):
        plt.text(j, i, f"{z2.values[i * len(embed_dim) + j]:.4f}", ha='center', va='center', color='black' if z2.values[i * len(embed_dim) + j] > colorVal else "white")

plt.tight_layout()
plt.savefig("AttentionTime_2Heads_PostNorm.png", bbox_inches='tight', dpi=400)
plt.clf()
