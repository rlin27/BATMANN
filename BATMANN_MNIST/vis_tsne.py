import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

exp_id = 2

num = 30

features = np.load(r'C:\Users\ruili\Rui\05_Codes\BATMANN\BATMANN_MNIST' + r'\ks-exp' + str(exp_id) + '.npy')  # [#samples, #dim]
label = np.load(r'C:\Users\ruili\Rui\05_Codes\BATMANN\BATMANN_MNIST' + r'\vs-exp' + str(exp_id) + '.npy')  # [#samples, labels]
print(label)

label_text = [str(x) for x in label]
features_embedded = TSNE(n_components=2).fit_transform(features)
plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=label)
for i in range(30):
    plt.text(features_embedded[i, 0], features_embedded[i, 1], label_text[i])
plt.savefig(r'C:\Users\ruili\Rui\05_Codes\BATMANN\BATMANN_MNIST' + r'\exp' + str(exp_id) + '.png', dpi=1000)
plt.show()