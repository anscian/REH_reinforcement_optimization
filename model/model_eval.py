from model import PropertyPredictor
from pprint import pprint
import torch
import matplotlib.pyplot as plt


checkpoint = torch.load('best.pth')

pprint(checkpoint['config'])

model = PropertyPredictor(checkpoint['config'])
print(model.eval())

model.load_state_dict(checkpoint['model_state_dict'])
print(model.test_model())

y_train_act, y_train_pred = [], []
for img_tensor, scaler_tensor, target_tensor in model.train_loader:
    y_train_act.append(target_tensor)
    with torch.no_grad():
        y_train_pred.append(model(img_tensor, scaler_tensor))

y_test_act, y_test_pred = [], []
for img_tensor, scaler_tensor, target_tensor in model.test_loader:
    y_test_act.append(target_tensor)
    with torch.no_grad():
        y_test_pred.append(model(img_tensor, scaler_tensor))

y_train_act, y_train_pred, y_test_act, y_test_pred = map(lambda x : torch.cat(x, dim=0), (y_train_act, y_train_pred, y_test_act, y_test_pred))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titles = ['E/Es', 'v']

for i in range(2):
    ax = axes[i]
    ax.scatter(y_train_act[:, i], y_train_pred[:, i], 
               alpha=0.5, label="Train", s=10)

    ax.scatter(y_test_act[:, i], y_test_pred[:, i], 
               alpha=0.5, label="Test", s=10)

    min_val = min(
        y_train_act[:, i].min(), y_test_act[:, i].min()
    ).item()
    max_val = max(
        y_train_act[:, i].max(), y_test_act[:, i].max()
    ).item()

    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(titles[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()