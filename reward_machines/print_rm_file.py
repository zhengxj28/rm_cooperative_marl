from itertools import permutations

########## minecraft2/nav_map5/navigationteam #########

# for p in permutations([1,2,3,4,5]):
#     i1,i2,i3,i4,i5 = p
#     print("(0, 1, ('a%d','b%d','c%d','d%d','e%d'), 1)"%(i1,i2,i3,i4,i5))

########## pass_room/4button3agent/passteam #########
# r_neg, r_mid, r_plus = -0.1, 0.1, 1
# for p in permutations([1, 2, 3]):
#     i, j, k = p
#     print("(0, 1, ('a%d','b%d'), %.1f)" % (i, j, r_mid))
#     print("(0, 1, ('a%d','b%d','r%d'), %.1f)" % (i, j, k, r_mid))
#
#     print("(1, 2, ('a%d','c%d', 'r%d'), %.1f)" % (i, k, k, r_mid))
#     print("(1, 2, ('b%d','c%d', 'r%d'), %.1f)" % (j, k, k, r_mid))
#     print("(1, 2, ('a%d','d%d', 'r%d'), %.1f)" % (i, k, k, r_mid))
#     print("(1, 2, ('b%d','d%d', 'r%d'), %.1f)" % (j, k, k, r_mid))
#
#     print("(2, 3, ('c%d','d%d', 'r%d', 'r%d'), %.1f)" % (k, i, k, i, r_mid))
#     print("(2, 3, ('c%d','d%d', 'r%d', 'r%d'), %.1f)" % (k, j, k, j, r_mid))
#
#     print("(3, 4, ('c%d','d%d','r1','r2','r3',), %.1f)" % (i, j, r_plus))
#
#     print("(2, 1, ('a%d','b%d'), %.1f)" % (i, j, r_neg))
#     print("(3, 2, ('c%d','r%d','r%d'), %.1f)" % (i, i, j, r_neg))
#     print("(3, 2, ('d%d','r%d','r%d'), %.1f)" % (i, i, j, r_neg))
#
# for i in range(1, 4):
#     print("(0, 1, ('r%d',), %.1f)" % (i, r_mid))
#     print("(1, 0, ('a%d',), %.1f)" % (i, r_neg))
#     print("(1, 0, ('b%d',), %.1f)" % (i, r_neg))
#     print("(2, 1, ('a%d'), %.1f)" % (i, r_neg))
#     print("(2, 1, ('b%d'), %.1f)" % (i, r_neg))
#     print("(3, 4, ('c%d','r1','r2','r3',), %.1f)" % (i, r_plus))
#     print("(3, 4, ('d%d','r1','r2','r3',), %.1f)" % (i, r_plus))
#     print("(3, 2, ('c%d','r%d'), %.1f)" % (i, i, r_neg))
#     print("(3, 2, ('d%d','r%d'), %.1f)" % (i, i, r_neg))
#
# print("(1, 0, (), %.1f)" % r_neg)
# print("(2, 1, (), %.1f)" % r_neg)
# print("(3, 4, ('r1','r2','r3',), %.1f)" % r_plus)

########## pass_room/4button3agent/passteam2 #########
# r_neg, r_mid, r_plus = -0.1, 0.1, 1
# for p in permutations([1, 2, 3]):
#     i, j, k = p
#     print("(0, 1, ('a%d','b%d','r%d'), %.1f)" % (i, j, k, r_mid))
#
#     print("(1, 2, ('a%d','r%d', 'c%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
#     print("(1, 2, ('a%d','r%d', 'd%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
#     print("(1, 2, ('r%d','b%d', 'c%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
#     print("(1, 2, ('r%d','b%d', 'd%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
#
#     print("(2, 3, ('c%d','d%d', 'r%d', 'r%d', 'r%d'), %.1f)" % (i, j, i, j, k, r_plus))
#
#     print("(1, 0, ('a%d','b%d'), %.1f)" % (i, j, r_neg))
#     print("(1, 0, ('a%d'), %.1f)" % (i, r_neg))
#     print("(1, 0, ('b%d'), %.1f)" % (i, r_neg))
#     print("(1, 0, (), %.1f)" % r_neg)
#
#     print("(2, 1, ('a%d','b%d','r%d'), %.1f)" % (i, j, k, r_neg))
#     print("(2, 1, ('a%d','r%d'), %.1f)" % (i, k, r_neg))
#     print("(2, 1, ('b%d','r%d'), %.1f)" % (i, k, r_neg))
#     print("(2, 1, ('r%d'), %.1f)" % (k, r_neg))
#
#     print("(2, 1, ('a%d','b%d','c%d','r%d'), %.1f)" % (i, j, k, k, r_neg))
#     print("(2, 1, ('a%d','c%d','r%d'), %.1f)" % (i, k, k, r_neg))
#     print("(2, 1, ('b%d','c%d','r%d'), %.1f)" % (i, k, k, r_neg))
#     print("(2, 1, ('r%d''c%d',), %.1f)" % (k, k, r_neg))
#
#     print("(2, 1, ('a%d','b%d','d%d','r%d'), %.1f)" % (i, j, k, k, r_neg))
#     print("(2, 1, ('a%d','d%d','r%d'), %.1f)" % (i, k, k, r_neg))
#     print("(2, 1, ('b%d','d%d','r%d'), %.1f)" % (i, k, k, r_neg))
#     print("(2, 1, ('r%d''d%d',), %.1f)" % (k, k, r_neg))

########## pass_room/4button3agent/passteam4#########
r_neg, r_mid, r_plus = 0, 0, 1
for p in permutations([1, 2, 3]):
    i, j, k = p
    print("(0, 1, ('a%d','b%d','r%d'), %.1f)" % (i, j, k, r_mid))

    print("(1, 2, ('a%d','r%d', 'c%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
    print("(1, 2, ('a%d','r%d', 'd%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
    print("(1, 2, ('r%d','b%d', 'c%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))
    print("(1, 2, ('r%d','b%d', 'd%d', 'r%d'), %.1f)" % (i, j, k, k, r_mid))

    print("(2, 3, ('c%d','d%d', 'r%d', 'r%d', 'r%d'), %.1f)" % (i, j, i, j, k, r_plus))
