import os
import html
import pickle
import xml.etree.ElementTree as ET

from scipy.sparse import csr_matrix


def save_data(data_counter, documents, labels):
    """
    Pickles the documents and labels array before deletion (to save memory).
    """
    global pkl_counter
    pickle.dump(documents, open(out_dir + "/documents-{}-{}.pkl".format(pkl_counter, data_counter), "wb"))
    pickle.dump(csr_matrix(labels),
                open(out_dir + "/labels-{}-{}.pkl".format(pkl_counter, data_counter), "wb"))
    pkl_counter += 1

# Counter for pickled files
pkl_counter = 0

# Make directories for pickles
out_dir = os.path.abspath(os.path.join(os.path.curdir, "data", "RCV1", "pickles", "RCV1-v2_Sparse"))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Get list of categories
categories = list(open("data/RCV1/RCV1_Uncompressed/appendices/rcv1.topics.txt", "r", encoding="utf-8")
                  .read().splitlines())
pickle.dump(categories, open(out_dir + "/class_names.pkl", "wb"))

# Get list of RCV1-v2 IDs
valid_ids = list(open("data/RCV1/RCV1_Uncompressed/appendices/rcv1v2-ids.dat", "r", encoding="utf-8")
                 .read().splitlines())

# Get mapping of IDs to categories
_item_categories = list(open("data/RCV1/RCV1_Uncompressed/appendices/rcv1-v2.topics.qrels", "r",
                             encoding="utf-8").read().splitlines())
_item_categories = [i.split() for i in _item_categories]
item_categories = {}
for line in _item_categories:
    if line[1] in item_categories:
        item_categories[line[1]].append(line[0])
    else:
        item_categories[line[1]] = [line[0]]

# Get list of directories in uncompressed dataset
uncompressed = os.listdir("data/RCV1/RCV1_Uncompressed")
dirs = list(filter(lambda x: x.startswith("1"), uncompressed))
dirs.sort()

data_counter = 0
documents = []
labels = []
for d in dirs:
    files = os.listdir("data/RCV1/RCV1_Uncompressed/" + d)
    files.sort()

    for f in files:
        tree = ET.parse("data/RCV1/RCV1_Uncompressed/" + d + "/" + f)
        root = tree.getroot()

        # Check for valid ID according to RCV1-v2
        if root.attrib["itemid"] not in valid_ids:
            continue

        # Get headline of news item
        headline = root.findall("headline")
        assert len(headline) == 1
        headline = headline[0].text
        if headline is None:
            headline = ""

        # Get content of news item
        content = root.findall("text")
        assert len(content) == 1
        content = content[0]
        text = ""
        for p in content.findall("p"):  # only <p> child elements
            text += " " + html.unescape(p.text)

        # Concatenate headline + " " + content to form document
        document = headline + text

        # Get categories of news item from XML file
        if False:
            codes = root.findall("./metadata/codes[@class='bip:topics:1.0']")
            if len(codes) == 0:
                continue
            assert len(codes) == 1
            codes = codes[0]
            _labels = [0] * len(categories)
            for code in codes.findall("code"):
                _labels[categories.index(code.attrib["code"])] = 1

        # Get categories of news item from RCV1-v2 file
        if True:
            _labels = [0] * len(categories)
            for c in item_categories[root.attrib["itemid"]]:
                _labels[categories.index(c)] = 1

        documents.append(document)  # append document to documents array
        labels.append(_labels)  # append extracted categories to labels array

        data_counter += 1
        print("{} {}".format(data_counter, f))
        if data_counter % 100000 == 0:
            save_data(data_counter, documents, labels)
            del documents, labels
            documents = []
            labels = []

save_data(data_counter, documents, labels)
