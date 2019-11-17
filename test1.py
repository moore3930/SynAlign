import matplotlib.pyplot as plt
import os
import numpy as np

# w=10
# h=10
# fig=plt.figure(figsize=(8, 8))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()

# import seaborn as sns
plt.clf()
f = plt.figure(figsize=(32, 32))

col = 6
row = 6

# # add colorbar
# cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
# cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
# cbar.ax.set_xlabel('Probability', labelpad=2)

for i in range(1, row*col+1):
    ax = f.add_subplot(row, col, i)

    # add image
    i = ax.imshow(np.random.rand(np.random.randint(1, 5), np.random.randint(1, 5)), cmap='gray', aspect='equal')

    # add labels
    ax.set_yticks(range(3))
    ax.set_yticklabels(['1', '2', '3'])

    ax.set_xticks(range(3))
    ax.set_xticklabels(['1', '2', '3'])

    ax.set_xlabel('Target Sequence')
    ax.set_ylabel('Source Sequence')

    # ax.legend(loc='best')

HERE = os.path.realpath(os.path.join(os.path.realpath(__file__), '..'))
f.savefig(os.path.join(HERE, 'attention_maps', 'test' + '.pdf'), bbox_inches='tight')

