import json
f = open('level_1_classes.json')
#f = open('init_classes.json')
data = json.load(f)
print(data.keys())