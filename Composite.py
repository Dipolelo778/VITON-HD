# composite.py

def composite(person, cloth, mask):
    return person * (1 - mask) + cloth * mask
