from model import PropertyPredictor
from pprint import pprint
import torch
import matplotlib.pyplot as plt

run = 'no_scaler'
checkpoint = torch.load(f'runs/{run}/best.pth')

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

mae = lambda act, pred : torch.mean(torch.abs(act - pred))
r2 = lambda act, pred : 1 - torch.sum((act - pred)**2) / torch.sum((act - torch.mean(act))**2)
print('Train MAE: E/Es', mae(y_train_act[:, 0], y_train_pred[:, 0]), 'v', mae(y_train_act[:, 1], y_train_pred[:, 1]))
print('Test MAE: E/Es', mae(y_test_act[:, 0], y_test_pred[:, 0]), 'v', mae(y_test_act[:, 1], y_test_pred[:, 1]))
print('Train R2: E/Es', r2(y_train_act[:, 0], y_train_pred[:, 0]), 'v', r2(y_train_act[:, 1], y_train_pred[:, 1]))
print('Test R2: E/Es', r2(y_test_act[:, 0], y_test_pred[:, 0]), 'v', r2(y_test_act[:, 1], y_test_pred[:, 1]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titles = ['$E/E_s$', '$\\nu$']

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

    ax.set_xlabel("Actual " + titles[i])
    ax.set_ylabel("Predicted " + titles[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(f'runs/{run}/model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()