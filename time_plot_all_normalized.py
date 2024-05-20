import pandas as pd
import matplotlib.pyplot as plt

seq_len = (64,100,128,256,512)#,1024)
embed_dim = (32,64,128,256,512)#,1024)

# Load the CSV file
data1 = pd.read_csv('test_times_1heads_PostNorm.csv')
data2 = pd.read_csv('test_times_2heads_PostNorm.csv')
data4 = pd.read_csv('test_times_4heads_PostNorm.csv')

# Extract the columns
x = data1.iloc[:, 0]
y = data1.iloc[:, 1]
f = data1.iloc[:, 2]*1000
z1 = data1.iloc[:, 3]*1000
z2 = data2.iloc[:, 3]*1000
z3 = data4.iloc[:, 3]*1000

maxVal = max(f.max(), z1.max(), z2.max(), z3.max())
minVal = min(f.max(), z1.min(), z2.min(), z3.max())
colorVal = minVal + 0.3 * (maxVal - minVal)

# Create the heat maps
#plt.subplot(1, 2, 1)
plt.imshow(f.values.reshape((len(set(y)), len(set(x)))), cmap='hot', origin='lower', vmin=minVal, vmax=maxVal)
plt.colorbar()
plt.xlabel('Sequence Length')
plt.ylabel('Embedding Dimension')
plt.xticks(range(len(set(x))), seq_len)
plt.yticks(range(len(set(y))), embed_dim)
plt.title('Fourier Block Time (ms)')

print(len(x), len(y))

# Add value labels to the heat map
for i in range(len(seq_len)):
    for j in range(len(embed_dim)):
        plt.text(j, i, f"{f.values[i * len(embed_dim) + j]:.3f}", ha='center', va='center', color='black' if f.values[i * len(embed_dim) + j] > colorVal else "white")

plt.tight_layout()
plt.savefig("FourierTime_PostNorm_Normalized.png", bbox_inches='tight', dpi=400)
plt.clf()

#plt.subplot(1, 2, 2)
plt.imshow(z1.values.reshape((len(set(y)), len(set(x)))), cmap='hot', origin='lower', vmin=minVal, vmax=maxVal)
plt.colorbar()
plt.xlabel('Sequence Length')
plt.ylabel('Embedding Dimension')
plt.xticks(range(len(set(x))), seq_len)
plt.yticks(range(len(set(y))), embed_dim)
plt.title('1 Head Attention Block Time (ms)')

# Add value labels to the heat map
for i in range(len(seq_len)):
    for j in range(len(embed_dim)):
        plt.text(j, i, f"{z1.values[i * len(embed_dim) + j]:.3f}", ha='center', va='center', color='black' if z1.values[i * len(embed_dim) + j] > colorVal else "white")

plt.tight_layout()
plt.savefig("AttentionTime_1Heads_PostNorm_Normalized.png", bbox_inches='tight', dpi=400)
plt.clf()

plt.imshow(z2.values.reshape((len(set(y)), len(set(x)))), cmap='hot', origin='lower', vmin=minVal, vmax=maxVal)
plt.colorbar()
plt.xlabel('Sequence Length')
plt.ylabel('Embedding Dimension')
plt.xticks(range(len(set(x))), seq_len)
plt.yticks(range(len(set(y))), embed_dim)
plt.title('2 Heads Attention Block Time (ms)')

# Add value labels to the heat map
for i in range(len(seq_len)):
    for j in range(len(embed_dim)):
        plt.text(j, i, f"{z2.values[i * len(embed_dim) + j]:.3f}", ha='center', va='center', color='black' if z2.values[i * len(embed_dim) + j] > colorVal else "white")

plt.tight_layout()
plt.savefig("AttentionTime_2Heads_PostNorm_Normalized.png", bbox_inches='tight', dpi=400)
plt.clf()

plt.imshow(z3.values.reshape((len(set(y)), len(set(x)))), cmap='hot', origin='lower', vmin=minVal, vmax=maxVal)
plt.colorbar()
plt.xlabel('Sequence Length')
plt.ylabel('Embedding Dimension')
plt.xticks(range(len(set(x))), seq_len)
plt.yticks(range(len(set(y))), embed_dim)
plt.title('4 Heads Attention Block Time (ms)')

# Add value labels to the heat map
for i in range(len(seq_len)):
    for j in range(len(embed_dim)):
        plt.text(j, i, f"{z3.values[i * len(embed_dim) + j]:.3f}", ha='center', va='center', color='black' if z3.values[i * len(embed_dim) + j] > colorVal else "white")

plt.tight_layout()
plt.savefig("AttentionTime_4Heads_PostNorm_Normalized.png", bbox_inches='tight', dpi=400)
plt.clf()