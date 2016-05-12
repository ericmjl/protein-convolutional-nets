"""
To use this script, type:

    $ python visualize_preds_actual.py [handle]

Of course, replace [handle] with the common prefix of the pickled data.
"""

import matplotlib.pyplot as plt
import pickle as pkl
import sys
import seaborn as sns

sns.set_context('poster')

from graphfp.utils import y_equals_x

if __name__ == '__main__':
    handle = sys.argv[1]
    animate = eval(sys.argv[2])

    with open('{0}_predsactual.pkl'.format(handle), 'rb') as f:
        preds_vs_actual = pkl.load(f)

    with open('{0}_trainloss.pkl'.format(handle), 'rb') as f:
        trainloss = pkl.load(f)

    with open('{0}_predsactual_cv.pkl'.format(handle), 'rb') as f:
        preds_vs_actual_cv = pkl.load(f)

    with open('{0}_trainloss_cv.pkl'.format(handle), 'rb') as f:
        trainloss_cv = pkl.load(f)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.yaxis.tick_right()

    if animate:
        plt.show(block=False)

        for i, data in sorted(preds_vs_actual.items(), key=lambda x: x[0]):
            if i % 1000 == 0:
                ax1.clear()
                ax1.scatter(data['preds'], data['actual'], color='blue')
                ax1.scatter(preds_vs_actual_cv[i]['preds'],
                            preds_vs_actual_cv[i]['actual'], color='red')
                ax1.set_title('Iteration {0}'.format(i))
                ax1.set_xlabel('predictions')
                ax1.set_ylabel('actual')
                ax1.plot(y_equals_x(data['actual']),
                         y_equals_x(data['actual']),
                         marker='o', ls='-', lw=2, color='red', alpha=0.2)

                ax2.clear()
                ax2.set_xlim(0, len(trainloss))
                ax2.plot(trainloss[:i], color='blue', label='train')
                ax2.plot(trainloss_cv[:i], color='red', label='test')
                ax2.set_title('Iteration {0}'.format(i))
                ax2.set_xlabel('iteration number')
                ax2.set_ylabel('training error')
                ax2.set_yscale('log')
                ax2.legend()

                plt.draw()
                plt.pause(1/100)
        plt.savefig('{0}_preds_trainloss.pdf'.format(handle),
                    bbox_inches='tight')
        plt.show(block=True)

    else:
        li = max(preds_vs_actual.keys())
        

        ax1.scatter(preds_vs_actual[li]['preds'], preds_vs_actual[li]['actual'], color='blue')
        ax1.scatter(preds_vs_actual_cv[li]['preds'],
                    preds_vs_actual_cv[li]['actual'], color='red')
        ax1.set_xlabel('predictions')
        ax1.set_ylabel('actual')
        ax1.set_title('convnet')

        ax2.set_xlim(0, len(trainloss))
        ax2.plot(trainloss[:li], color='blue', label='train')
        ax2.plot(trainloss_cv[:li], color='red', label='test')
        ax2.set_xlabel('iteration number')
        ax2.set_ylabel('training error')
        ax2.set_yscale('log')
        ax2.set_title('train error')
        ax2.legend()

        plt.subplots_adjust(bottom=0.2)
        plt.savefig('{0}_preds_trainloss.pdf'.format(handle),
                    bbox_inches='tight')

