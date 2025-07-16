# composite.py

def composite(warped_cloth, person, mask):
    return warped_cloth * mask + person * (1 - mask)
