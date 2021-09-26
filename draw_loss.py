import re
import matplotlib.pyplot as plt


with open('train_mlm.log','r') as fp:
    losses = []
    lines = fp.read().strip().split('\n')
    for line in lines:
        loss = re.search('loss:([0-9.]+)', line)
        if loss:
            loss = loss.groups()[0]
            losses.append("{:.4f}".format(float(loss)))
y_data = []
x_data = []
for i,loss in enumerate(losses):
    if i % 1000 == 0:
        print(loss)
        y_data.append(float(loss))
        x_data.append(i)
plt.xlabel('steps') #x轴表示
plt.ylabel('loss') #y轴表示
plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='--')
plt.savefig('loss.png')
plt.show()
