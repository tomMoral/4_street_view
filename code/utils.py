
def display_char(L, c, w, h, im, col='blue'):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    plt.clf()
    ax = plt.gca()
    for rect in L[c]:
        ax.add_patch(Rectangle(rect[0], w, h, fc='none', ec=col))
    ax.autoscale_view()
    plt.imshow(im)
