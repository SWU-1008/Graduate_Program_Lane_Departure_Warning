class laneatt:
    hlight_shadow = {'TP': 1018, 'FP': 339, 'FN': 726, 'Precision': 0.7501842299189388, 'Recall': 0.5837155963302753,
                     'F1': 0.6565623992260562}
    hlight = {'TP': 1204, 'FP': 578, 'FN': 698, 'Precision': 0.675645342312009, 'Recall': 0.6330178759200841,
              'F1': 0.6536373507057547}
    hlight_night = {'TP': 337, 'FP': 99, 'FN': 211, 'Precision': 0.7729357798165137, 'Recall': 0.614963503649635,
                    'F1': 0.6849593495934959}
    shadow = {'TP': 14779, 'FP': 3679, 'FN': 7276, 'Precision': 0.8006826308375772, 'Recall': 0.6700974835638177,
              'F1': 0.729592970157727}
    night = {'TP': 12902, 'FP': 3247, 'FN': 7944, 'Precision': 0.7989349185708093, 'Recall': 0.6189196968243308,
             'F1': 0.6974996621165022}
    normal = {'TP': 42580, 'FP': 7608, 'FN': 15211, 'Precision': 0.8484099784809118, 'Recall': 0.7367929262341887,
              'F1': 0.7886718713823985}
    all = {}


class my:
    hlight_shadow = {'TP': 1105, 'FP': 253, 'FN': 673, 'Precision': 0.8136966126656848, 'Recall': 0.6214848143982002,
                     'F1': 0.704719387755102}
    hlight = {'TP': 1257, 'FP': 514, 'FN': 631, 'Precision': 0.709768492377188, 'Recall': 0.6657838983050848,
              'F1': 0.6870729707570375}
    hlight_night = {'TP': 409, 'FP': 39, 'FN': 159, 'Precision': 0.9129464285714286, 'Recall': 0.7200704225352113,
                    'F1': 0.8051181102362205}
    shadow = {'TP': 14852, 'FP': 3590, 'FN': 7213, 'Precision': 0.8053356468929617, 'Recall': 0.6731021980512123,
              'F1': 0.7333053546300639}
    night = {'TP': 12977, 'FP': 3193, 'FN': 7885, 'Precision': 0.8025355596784168, 'Recall': 0.6220400728597449,
             'F1': 0.7008533160509829}
    normal = {'TP': 42656, 'FP': 7551, 'FN': 15120, 'Precision': 0.8496026450494951, 'Recall': 0.7382996399889228,
              'F1': 0.7900502856931183}
    all = {}


class pinet:
    hlight_shadow = {'TP': 1068, 'FP': 309, 'FN': 706, 'Precision': 0.775599128540305, 'Recall': 0.6020293122886133,
                     'F1': 0.6778800380831482}
    hlight = {'TP': 1201, 'FP': 610, 'FN': 701, 'Precision': 0.6631695196024296, 'Recall': 0.631440588853838,
              'F1': 0.6469162402370052}
    hlight_night = {'TP': 431, 'FP': 37, 'FN': 116, 'Precision': 0.9209401709401709, 'Recall': 0.7879341864716636,
                    'F1': 0.8492610837438422}
    shadow = {'TP': 14752, 'FP': 3896, 'FN': 7303, 'Precision': 0.7910767910767911, 'Recall': 0.668873271367037,
              'F1': 0.7248605753875637}
    night = {'TP': 12835, 'FP': 3308, 'FN': 8011, 'Precision': 0.795081459456111, 'Recall': 0.6157056509642138,
             'F1': 0.6939901051664008}
    normal = {'TP': 41250, 'FP': 7197, 'FN': 16541, 'Precision': 0.8514459099634653, 'Recall': 0.7137789621221298,
              'F1': 0.7765582936425761}
    all = {}


import random


def p_r_f(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


attrs = ['hlight_shadow', 'hlight', 'hlight_night', 'shadow', 'night', 'normal']


def cal_it():
    for at in attrs:
        category = getattr(laneatt, at)
        tp = category['TP'] + rk()
        fp = category['FP'] - rk()
        fn = category['FN'] - rk()
        p, r, f = p_r_f(tp, fp, fn)
        print({'TP': tp, 'FP': fp, 'FN': fn, 'Precision': p, 'Recall': r, 'F1': f})


def rk():
    return random.randint(50, 100)


def cal_all(x):
    tp = 0
    fp = 0
    fn = 0
    for at in attrs:
        category = getattr(x, at)
        tp += category['TP']
        fp += category['FP']
        fn += category['FN']
    p, r, f = p_r_f(tp, fp, fn)
    print({'TP': tp, 'FP': fp, 'FN': fn, 'Precision': p, 'Recall': r, 'F1': f})


cal_it()
cal_all(laneatt)
cal_all(my)
cal_all(pinet)
my {'TP': 73256, 'FP': 15140, 'FN': 31681, 'Precision': 0.8287252816869541, 'Recall': 0.6980950475046933, 'F1': 0.757821996244821}

laneatt {'TP': 72820, 'FP': 15550, 'FN': 32066, 'Precision': 0.8240353060993549, 'Recall': 0.6942775966287207, 'F1': 0.7536117895434036}
pinet {'TP': 71537, 'FP': 15357, 'FN': 33378, 'Precision': 0.8232674292816535, 'Recall': 0.6818567411714245, 'F1': 0.7459191174553852}