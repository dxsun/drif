from collections import defaultdict
import random, math

landmarks = {'Well', 'LionStatue', 'StreetLamp', 'WaterWell', 'Bench', 'Column', 'OilDrum', 'Tombstone', 'PhoneBox', 'Banana', 'Palm3', 'Apple', 'House', 'GoldCone', 'Anvil', 'SteelCube', 'TvTower', 'YellowFlowers', 'Box', 'BushTree', 'Barrel', 'BushTree2', 'RecycleBin', 'Soldier', 'Beacon', 'Stone3', 'LowPolyTree', 'Palm2', 'Stone1', 'Windmill', 'ConniferCluster', 'Tank', 'Pumpkin', 'Stone2', 'Tower2', 'Gorilla', 'TreasureChest', 'LpPine', 'BushTree3', 'Jet', 'Rock', 'House2', 'BigHouse', 'GiantPalm', 'TrafficCone', 'WoodBarrel', 'Coach', 'Pickup', 'Cactus', 'RedFlowers', 'WoodenChair', 'Ladder', 'Stump', 'House1', 'Barrel2', 'FireHydrant', 'Boletus', 'Container', 'Mushroom', 'Boat', 'Dumpster', 'Pillar', 'Palm1'}
min_zpos = 0
max_zpos = 1000
min_xpos = 0
max_xpos = 1000

"""
Note that the coordinates are (zPos, xPos) = (x, y) on the matplotlib plot

Assumptions/shortcuts in this config generation
- if the instruction contains "to the left of/to the right of", 
they will have a shifted zPos, but their xPos will be the same

- if there's ever a big/small object, there exists another of the same
object of the opposite size (they always come in pairs)

- regular size is always 75
"""
sizes = [50, 75, 100]

def opposite_size(size):
    if size == "big":
        return "small"
    elif size == "small":
        return "big"
    return None

class Config(object):
    def __init__(self, sizing='absolute'):
        """
        sizing is either 'absolute' or 'relative'. 
        - If 'absolute', 'small' denotes radius 50 and 'big' denotes radius 100. 
        - If 'relative', small and big can be chosen from (50, 75, 100)
        """
        self.config = {"zPos":[], "xPos":[], "isEnabled":[], "radius":[], "landmarkName":[], "goalLandmarkIndex":[]}

        self.sizing = sizing
        if sizing == 'relative':
            bottom_size_index = random.randint(0, 1)
            top_size_index = random.randint(bottom_size_index + 1, 2)

            self.relative_size_pair = sizes[bottom_size_index], sizes[top_size_index]

    def get_rand_pos(self, filler):
        if filler:
            pos = (int(random.random()*800 + 100), int(random.random()*800 + 100))
            return pos
        else:
            return (int(random.random()*600 + 200), int(random.random()*600 + 200))
            

    def add_objects(self, objects, pos=None, sizes=None, goal=False, filler=False):
        """
            filler - whether the object is part of the instruction, or 
            just a filler object in the background
        """
        if sizes:
            assert len(sizes) == len(objects), "Need to provide one size per added object"
        else:
            sizes = [None]

        # If the object is sized, add two of the same object, but with different sizes
        goal_landmark_indices = []
        for index, obj in enumerate(objects):
            placed_position = self.add_sized_object(obj, pos=pos, size=sizes[index], filler=filler)
            if goal:
                goal_landmark_indices.append(len(self.config["zPos"]) - 1)
            if sizes[index] in ["big", "small"]:
                self.add_sized_object(obj, size=opposite_size(sizes[index]), filler=filler)

        if goal:
            self.config["goalLandmarkIndex"].append(goal_landmark_indices)

        return placed_position

    def add_sized_object(self, obj, pos=None, size=None, filler=False):
        if pos is None:
            # pick a random non-taken spot
            count = 0
            while True:
                count += 1
                pos = self.get_rand_pos(filler=filler)

                viable_pos = True

                for i in range(len(self.config['zPos'])):
                    # If two objects are too close (z and x are both less than 150 away)
                    if abs(pos[0] - self.config['zPos'][i]) <= 200 and abs(pos[1] - self.config['xPos'][i]) <= 200:
                        viable_pos = False

                if viable_pos:
                    break
                
                if count > 5000:
                    return None

        self.config['zPos'].append(pos[0])
        self.config['xPos'].append(pos[1])
        self.config['isEnabled'].append(True)

        if self.sizing == "absolute":
            if size == "small":
                self.config['radius'].append(sizes[0])
            elif size == "big":
                self.config['radius'].append(sizes[2])
            else:
                self.config['radius'].append(sizes[1])
        elif self.sizing == "relative":
            if size == "small":
                self.config['radius'].append(self.relative_size_pair[0])
            elif size == "big":
                self.config['radius'].append(self.relative_size_pair[1])
            else:
                self.config['radius'].append(75)
        else:
            raise Exception("you need to pick either absolute or relative sizing")

        self.config['landmarkName'].append(obj)

        return pos


class CFG(object):
    def __init__(self, sizing='absolute'):
        self.prod = defaultdict(list)
        self.config = Config(sizing=sizing)

    def add_prod(self, lhs, rhs):
        """ Add production to the grammar. 'rhs' can
            be several productions separated by '|'.
            Each production is a sequence of symbols
            separated by whitespace.

            Usage:
                grammar.add_prod('NT', 'VP PP')
                grammar.add_prod('Digit', '1|2|3|4')
        """
        prods = rhs.split('|')
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))

    def weighted_choice(self, weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    def gen_random_convergent(
        self,
        symbol,
        cfactor=0.20,
        pcount=defaultdict(int)):
        """ Generate a random sentence from the
          grammar, starting with the given symbol.

          Uses a convergent algorithm - productions
          that have already appeared in the
          derivation on each branch have a smaller
          chance to be selected.

          cfactor - controls how tight the
          convergence is. 0 < cfactor < 1.0

          pcount is used internally by the
          recursive calls to pass on the
          productions that have been used in the
          branch.
        """
        sentence = ''

        # The possible productions of this symbol are weighted
        # by their appearance in the branch that has led to this
        # symbol in the derivation
        #
        weights = []
        for prod in self.prod[symbol]:
            if prod in pcount:
                weights.append(cfactor ** (pcount[prod]))
            else:
                weights.append(1.0)

        rand_prod = self.prod[symbol][self.weighted_choice(weights)]

        # pcount is a single object (created in the first call to
        # this method) that's being passed around into recursive
        # calls to count how many times productions have been
        # used.
        # Before recursive calls the count is updated, and after
        # the sentence for this call is ready, it is rolled-back
        # to avoid modifying the parent's pcount.
        #
        pcount[rand_prod] += 1


        print("prod:", rand_prod)

        # For the outermost production (the first VVi), we create the config based on the
        # structure of the produced sentence by slightly hard-coding it
        if rand_prod == ('VVi', 'ACT', 'DEST'):
            # take last word of the sentence for DEST and make that object appear
            for sym in rand_prod:
                # for non-terminals, recurse
                if sym in self.prod:
                    new_sentence_part = self.gen_random_convergent(
                        sym,
                        cfactor=cfactor,
                        pcount=pcount)

                    sentence += new_sentence_part

                    if sym == 'DEST':
                        split_sentence = new_sentence_part.split()
                        dest_obj = split_sentence[-1]
                        size = split_sentence[-2]
                        self.config.add_objects([dest_obj], sizes=[size], goal=True)
                else:
                    sentence += sym + ' '
        elif rand_prod == ('VVi', 'ACT', 'DEST', 'ACT', 'DESC'):
            # take last word of the sentence for DEST and make that object appear
            # take last word of sentence for DESC and make that object appear relation to DEST obj
            dest_pos = None
            for sym in rand_prod:
                if sym in self.prod:
                    new_sentence_part = self.gen_random_convergent(
                        sym,
                        cfactor=cfactor,
                        pcount=pcount)

                    sentence += new_sentence_part

                    if sym == 'DEST':
                        split_sentence = new_sentence_part.split()
                        dest_obj = split_sentence[-1]
                        dest_size = split_sentence[-2]
                        dest_pos = self.config.add_objects([dest_obj], sizes=[dest_size], goal=True)

                    elif sym == 'DESC':
                        desc_sent = new_sentence_part.split()
                        obj = desc_sent[-1]
                        size = desc_sent[-2]
                        relation = desc_sent[1]
                        if relation == "right":
                            pos = (dest_pos[0] - 200, dest_pos[1])

                            distractor_pos = (pos[0] - 200, dest_pos[1])
                        else:
                            pos = (dest_pos[0] + 200, dest_pos[1])

                            distractor_pos = (pos[0] + 200, dest_pos[1])
                        self.config.add_objects([obj], pos=pos, sizes=[size])

                        self.config.add_objects([dest_obj], pos=distractor_pos, sizes=[dest_size])

                else:
                    sentence += sym + ' '
        elif rand_prod == ('VVi', 'between', 'DP', 'and', 'DP'):
            for sym in rand_prod:
                if sym in self.prod:
                    new_sentence_part = self.gen_random_convergent(
                        sym,
                        cfactor=cfactor,
                        pcount=pcount)

                    sentence += new_sentence_part
                else:
                    sentence += sym + ' '

            _, two_objects = sentence.split(' between ')
            dp1, dp2 = two_objects.split(' and ')
            obj1, size1 = dp1.split()[-1], dp1.split()[-2]
            obj2, size2 = dp2.split()[-1], dp2.split()[-2]

            self.config.add_objects([obj1, obj2], sizes=[size1, size2], goal=True)

        else:
            for sym in rand_prod:
                if sym in self.prod:
                    sentence += self.gen_random_convergent(
                        sym,
                        cfactor=cfactor,
                        pcount=pcount)
                else:
                    sentence += sym + ' '

        # backtracking: clear the modification to pcount
        pcount[rand_prod] -= 1
        return sentence


"""
ROOT -> VP
VP -> VP 'before' VP
VP -> VVi ACT DEST
VP -> VVi ACT DEST ACT DESC
VP -> VVi 'between' DP 'and' DP
DEST -> DP
DEST -> DESC
DP -> 'the' NP
DESC -> 'the' CARD 'of' DP
NP -> NN
NP -> ADJ NN
VVi -> 'go' | 'fly'
NN -> 'Well' | 'LionStatue' | 'StreetLamp' | 'WaterWell'
ADJ -> 'red' | 'green' | 'blue' | 'big' | 'small'
CARD -> 'left' | 'right'
ACT -> 'to' | 'over'
"""

all_objects = "Well | LionStatue | StreetLamp | WaterWell | Bench | Column | OilDrum | Tombstone | PhoneBox | Banana | Palm3 | Apple | House | GoldCone | Anvil | SteelCube | TvTower | YellowFlowers | Box | BushTree | Barrel | BushTree2 | RecycleBin | Soldier | Beacon | Stone3 | LowPolyTree | Palm2 | Stone1 | Windmill | ConniferCluster | Tank | Pumpkin | Stone2 | Tower2 | Gorilla | TreasureChest | LpPine | BushTree3 | Jet | Rock | House2 | BigHouse | GiantPalm | TrafficCone | WoodBarrel | Coach | Pickup | Cactus | RedFlowers | WoodenChair | Ladder | Stump | House1 | Barrel2 | FireHydrant | Boletus | Container | Mushroom | Boat | Dumpster | Pillar | Palm1"

cfg = CFG(sizing='relative')
cfg.add_prod('ROOT', 'VP')
cfg.add_prod('VP', 'VP before VP')
cfg.add_prod('VP' , 'VVi ACT DEST')
cfg.add_prod('VP' , 'VVi ACT DEST ACT DESC')
cfg.add_prod('VP' , 'VVi between DP and DP')
cfg.add_prod('DEST' , 'DP')
cfg.add_prod('DEST' , 'DESC')
cfg.add_prod('DP' , 'the NP')
cfg.add_prod('DESC' , 'the CARD of DP')
cfg.add_prod('NP' , 'NN')
cfg.add_prod('NP' , 'ADJ NN')
cfg.add_prod('VVi' , 'go | fly')
cfg.add_prod('NN' , all_objects)
cfg.add_prod('ADJ' , 'big | small')
cfg.add_prod('CARD' , 'left | right')
cfg.add_prod('ACT' , 'to')


print("instruction:", cfg.gen_random_convergent('ROOT'))

config = cfg.config.config

# print(config)


# Add the random landmarks
num_extra_landmarks = 8 - len(config['landmarkName'])

for _ in range(num_extra_landmarks):
    while True:
        landmark = list(landmarks)[random.randint(0, len(landmarks) - 1)]
        if landmark in config['landmarkName']:
            continue
        break

    cfg.config.add_objects([landmark], filler=True)


# Final adjustments - if things exist out of bounds, just shift to the right or left
for index, zpos in enumerate(config['zPos']):
    if zpos - config['radius'][index] < 0:
        # shift everything right by config['radius'][index] - zpos
        for i in range(len(config['zPos'])):
            config['zPos'][i] += config['radius'][index] - zpos

for index, zpos in enumerate(config['zPos']):
    if zpos + config['radius'][index] > max_zpos:
        # shift everything left by config['radius'][index] + zpos - max_zpos
        for i in range(len(config['zPos'])):
            config['zPos'][i] -= config['radius'][index] + zpos - max_zpos

# Rotation
#
# Make (0,0) the center of everything, rotate by a random multiple of 90 degrees counterclockwise,
# then restore to normal coordinates
zPos = config['zPos']
xPos = config['xPos']
for i in range(len(zPos)):
    zPos[i] -= 500
    xPos[i] -= 500

index = random.randint(0, 3)
rotation_degree = index*math.pi/2

for i in range(len(zPos)):
    temp_zpos = zPos[i]*math.cos(rotation_degree) - xPos[i]*math.sin(rotation_degree)
    temp_xpos = zPos[i]*math.sin(rotation_degree) + xPos[i]*math.cos(rotation_degree)
    zPos[i] = temp_zpos
    xPos[i] = temp_xpos

for i in range(len(zPos)):
    zPos[i] = int(zPos[i] + 500)
    xPos[i] = int(xPos[i] + 500)
###

config['zPos'] = zPos
config['xPos'] = xPos
print('counterclockwise rotation degree:', rotation_degree*180/math.pi)

if index == 0:
    starting_pos = (random.randint(0, 1000), 0)
elif index == 1:
    starting_pos = (1000, random.randint(0, 1000))
elif index == 2:
    starting_pos = (random.randint(0, 1000), 1000)
elif index == 3:
    starting_pos = (0, random.randint(0, 1000))

print("starting pos:", starting_pos)

import matplotlib.pyplot as plt

markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd']
plotted_landmarks = set()
print("zpos:", config['zPos'])
print("xpos:", config['xPos'])
print("land:", config['landmarkName'])
for i in range(len(config['zPos'])):
    landmark = config['landmarkName'][i]
    if landmark not in plotted_landmarks:
        plotted_landmarks.add(landmark)
    else:
        landmark = landmark + "_other"
    plt.plot(config['zPos'][i], config['xPos'][i], markers[i], label="'{0}'".format(landmark))
    plt.legend(loc=(1.01,0), numpoints=1)
    plt.ylim(0, 1000)
    plt.xlim(0, 1000)

plt.show()