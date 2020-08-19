from matplotlib import pyplot as plt

class draw_efficient_curve:
    def __init__(self, target_hardware):
        self.target_hardware = target_hardware
        self.latency_list = []  # flops
        self.params_list = []

        self.top1s = []
        self.legend = []
        self.com = {}
        self.com_2 = {}

        # {'ResNetV1': ([1818.21, 3669.16, 3868.96, 7586.3], [71.6, 76.0, 78.6, 79.9]),
        #  'ResNetV2': ([1818.41, 3669.36, 4100.9, 7818.24], [72.5, 75.8, 78.0, 79.2])}
        # {'ResNetV1': ([11.15, 20.79, 24.39, 42.52], [71.6, 76.0, 78.6, 79.9]),
        #  'ResNetV2': ([11.15, 20.79, 24.37, 42.48], [72.5, 75.8, 78.0, 79.2])}

    def adj(self, latency_list, top1s):
        c = []
        d = []
        b = sorted(enumerate(latency_list), key=lambda x: x[1])
        for i in b:
            c.append(i[0])
        for i in c:
            d.append(top1s[i])
        return sorted(latency_list), d

    def update_com(self, style, latency, top1, creat_new):
        if creat_new:
            # if style == 'ResNetV1':
            #     self.com[style] =  ([1818.21, 3669.16, 3868.96, 7586.3], [71.6, 76.0, 78.6, 79.9]) #([1818.21, 3669.16, 11536.78], [71.6, 76.0, 79.21])
            # elif style == 'ResNetV2':
            #     self.com[style] = ([1818.41, 3669.36, 4100.9, 7818.24], [72.5, 75.8, 78.0, 79.2])
            # else:
            #     self.com[style] = ([], [])
            self.com[style] = ([], [])
        self.com[style][0].append(latency)
        self.com[style][1].append(top1)
        a, b = self.adj(self.com[style][0], self.com[style][1])
        self.com[style] = (a, b)
        return self.com

    def update_com_2(self, style, latency, top1, creat_new):
        if creat_new:
            # if style == 'ResNetV1':
            #     self.com_2[style] = ([11.15, 20.79, 24.39, 42.52], [71.6, 76.0, 78.6, 79.9]) # ([11.15, 20.79, 57.4], [71.6, 76.0, 79.21])
            # elif style == 'ResNetV2':
            #     self.com_2[style] = ([11.15, 20.79, 24.37, 42.48], [72.5, 75.8, 78.0, 79.2])
            # else:
            #     self.com_2[style] = ([], [])
            self.com_2[style] = ([], [])
        self.com_2[style][0].append(latency)
        self.com_2[style][1].append(top1)
        a, b = self.adj(self.com_2[style][0], self.com_2[style][1])
        self.com_2[style] = (a, b)
        return self.com_2

    def plot(self, style, latency, params, top1):
        creat_new = False
        if not style in self.legend:
            self.legend.append(style)
            creat_new = True

        self.com = self.update_com(style, latency, top1, creat_new)

        self.com_2 = self.update_com_2(style, params, top1, creat_new)



        plt.figure(figsize=(12, 4))
        ax0 = plt.subplot(1, 2, 1)
        # plt.figure(figsize=(4,4))
        for i in range(len(self.legend)):
            if i == 0:
                ax0.plot(self.com[self.legend[i]][0], self.com[self.legend[i]][1], 'x-', marker='*', color='darkred',
                         linewidth=2, markersize=8, label=style)
            elif i == 1:
                # print(self.com[self.legend[i]][0], self.com[self.legend[i]][1])
                ax0.plot(self.com[self.legend[i]][0], self.com[self.legend[i]][1], '--', marker='+', linewidth=2,
                         markersize=8, label=style)
            elif i == 2:
                # print(self.com)
                ax0.plot(self.com[self.legend[i]][0], self.com[self.legend[i]][1], '--', marker='>', linewidth=2,
                         markersize=8, label=style)

        plt.xlabel('%s MFLOPS' % self.target_hardware, size=12)
        plt.ylabel('Imagenet Top-1 Accuracy (%)', size=12)
        plt.legend(self.legend, loc='lower right')
        plt.grid(True)

        ax1 = plt.subplot(1, 2, 2)
        for i in range(len(self.legend)):
            if i == 0:
                ax1.plot(self.com_2[self.legend[i]][0], self.com_2[self.legend[i]][1], 'x-', marker='*',
                         color='darkred', linewidth=2, markersize=8, label=style)
            elif i == 1:
                # print(self.com[self.legend[i]][0], self.com[self.legend[i]][1])
                ax1.plot(self.com_2[self.legend[i]][0], self.com_2[self.legend[i]][1], '--', marker='+', linewidth=2,
                         markersize=8, label=style)
            elif i == 2:

                ax1.plot(self.com_2[self.legend[i]][0], self.com_2[self.legend[i]][1], '--', marker='>', linewidth=2,
                         markersize=8, label=style)

        plt.xlabel('%s Params (MB)' % self.target_hardware, size=12)
        plt.ylabel('Imagenet Top-1 Accuracy (%)', size=12)
        # print(style)
        plt.legend(self.legend, loc='lower right')
        plt.grid(True)

        plt.show()
        # print(self.com)
        # print(self.com_2)
        # print('Successfully draw the tradeoff curve!')
        # record.plot('XXX', flops, params, acc)