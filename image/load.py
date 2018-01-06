import turicreate as tc

def path2label(path):
    if "Border_terrier" in path:
        return "Border_terrier"
    elif "cairn" in path:
        return "cairn"
    elif "pug" in path:
        return "pug"
    elif "Shih-Tzu" in path:
        return "Shih-Tzu"
    else:
        return "unknown"

def gen_sframe():
    data = tc.image_analysis.load_images('../data/dog/', with_path=True)
    data['label'] = data['path'].apply(path2label)
    data.save('data/train.sframe')
    data.explore()

gen_sframe()