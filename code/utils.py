
def display_char(L, c, w, h, im, col='blue'):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    plt.clf()
    ax = plt.gca()
    for rect in L[c]:
        ax.add_patch(Rectangle(rect[0], w, h, fc='none', ec=col))
    ax.autoscale_view()
    plt.imshow(im)
    plt.show()

def extract_xml(fname, base_dir = '../data/'):
    from os.path.join import join as jp
    from xml.etree.ElementTree as ET
    tree = ET.parse(fname)
    root = tree.getroot()

    filenames = []
    labels = []

    for child in root:
        if child.get('tag') not in ['(',')','!','&','?',
                                    '.','"',"'",'-',':',
                                    u'\xa3', u'\xc9',',',
                                    u'\xd1', u'\xe9']:
            filenames.append(jp(base_dir, child.get('file')))
            labels.append(child.get('tag'))

    np.save('list_char', filenames)
    np.save('lab_char', labels)

